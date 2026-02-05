import hashlib
import os
import datetime
import numpy as np
import typer
import fiftyone as fo
import fiftyone.zoo as foz
from typing import List, Dict, Tuple, Set, Any
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

app = typer.Typer(rich_markup_mode="rich")

# --- Type Definitions ---
MatchResult = Dict[str, Any]  # Dictionary containing type, dist, paths, etc.
HashEntry = Dict[str, List[str]]  # MD5 -> List of filepaths


# --- Utility Functions ---

def get_file_hash(path: str) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception:
        return ""
    return hash_md5.hexdigest()


def format_path(full_path: str) -> str:
    """Formats path for display (last 3 parts)."""
    parts = Path(full_path).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else full_path


def parse_input_paths(paths: List[str]) -> List[str]:
    """Parses input arguments, handling space or semicolon delimiters."""
    if len(paths) == 1 and ";" in paths[0]:
        return [p.strip() for p in paths[0].split(";") if p.strip()]
    return paths


# --- Core Logic Functions ---

def load_and_process_dataset(
        path: str,
        index: int,
        model: Any
) -> Tuple[fo.Dataset, np.ndarray, HashEntry]:
    """
    Loads images from a directory into FiftyOne, computes hashes, and calculates embeddings.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        typer.secho(f"ERROR: Path does not exist: {abs_path}", fg=typer.colors.RED)
        return None, None, {}

    folder_name = os.path.basename(abs_path.rstrip(os.sep))
    ds_name = f"source_{index}_{folder_name}"

    # Cleanup existing dataset with same name
    if ds_name in fo.list_datasets():
        fo.delete_dataset(ds_name)

    typer.echo(f"Processing: {ds_name}...")
    ds = fo.Dataset.from_dir(
        dataset_dir=abs_path,
        dataset_type=fo.types.ImageDirectory,
        name=ds_name
    )
    ds.persistent = True

    # 1. Compute Hashes
    local_hashes: HashEntry = {}
    for sample in ds:
        h = get_file_hash(sample.filepath)
        if h:
            if h not in local_hashes:
                local_hashes[h] = []
            local_hashes[h].append(sample.filepath)

    # 2. Compute Embeddings
    typer.echo(f" [{ds_name}] images: {len(ds)}. Computing embeddings...")
    embeddings = ds.compute_embeddings(model=model)

    return ds, embeddings, local_hashes


def find_exact_matches(
        all_hashes: HashEntry
) -> Tuple[List[MatchResult], Set[Tuple[str, str]]]:
    """
    Identifies exact duplicates based on MD5 hash collisions.
    Returns the matches and a set of processed pairs to avoid redundancy.
    """
    matches = []
    processed_pairs = set()

    typer.secho("\nSearching for exact matches via hash...", fg=typer.colors.GREEN, bold=True)

    for h, file_list in all_hashes.items():
        if len(file_list) > 1:
            # Compare all files within the same hash group
            original = file_list[0]
            for duplicate in file_list[1:]:
                matches.append({
                    "type": "EXACT",
                    "dist": 0.0,
                    "pretty_new": format_path(duplicate),
                    "pretty_old": format_path(original),
                    "path_new": duplicate,
                    "path_old": original
                })
                # Store pair key
                pair_key = tuple(sorted([duplicate, original]))
                processed_pairs.add(pair_key)

    return matches, processed_pairs


def find_visual_matches(
        datasets: List[fo.Dataset],
        embeddings_list: List[np.ndarray],
        processed_pairs: Set[Tuple[str, str]],
        dup_thresh: float,
        sim_thresh: float
) -> List[MatchResult]:
    """
    Identifies visual duplicates using Nearest Neighbors on CLIP embeddings.
    """
    matches = []
    typer.secho("Searching for visual matches (CLIP)...", fg=typer.colors.MAGENTA, bold=True)

    # Compare every dataset against all previous datasets
    for i in range(1, len(datasets)):
        query_ds = datasets[i]
        query_emb_raw = embeddings_list[i]
        query_paths = query_ds.values("filepath")

        # Filter None embeddings
        valid_q_idxs = [idx for idx, e in enumerate(query_emb_raw) if e is not None]
        if not valid_q_idxs: continue

        query_emb = np.array([query_emb_raw[idx] for idx in valid_q_idxs])
        query_paths_v = [query_paths[idx] for idx in valid_q_idxs]

        for j in range(i):
            index_ds = datasets[j]
            index_emb_raw = embeddings_list[j]
            index_paths = index_ds.values("filepath")

            valid_i_idxs = [idx for idx, e in enumerate(index_emb_raw) if e is not None]
            if not valid_i_idxs: continue

            index_emb = np.array([index_emb_raw[idx] for idx in valid_i_idxs])
            index_paths_v = [index_paths[idx] for idx in valid_i_idxs]

            typer.echo(f"  Comparing: {query_ds.name} vs {index_ds.name}")

            # Vector Search
            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(index_emb)
            distances, indices = nn.kneighbors(query_emb)

            for k in range(len(query_emb)):
                dist = float(distances[k][0])
                if dist <= sim_thresh:
                    path_new = query_paths_v[k]
                    path_old = index_paths_v[indices[k][0]]

                    # Skip if already found via hash
                    if tuple(sorted([path_new, path_old])) in processed_pairs:
                        continue

                    m_type = "DUPLICATE" if dist <= dup_thresh else "SIMILAR"
                    matches.append({
                        "type": m_type,
                        "dist": dist,
                        "pretty_new": format_path(path_new),
                        "pretty_old": format_path(path_old),
                        "path_new": path_new,
                        "path_old": path_old
                    })
    return matches


def generate_text_report(
        matches: List[MatchResult],
        filename: str,
        dup_thresh: float,
        sim_thresh: float
):
    """Generates and saves a text report of the findings."""
    if not matches:
        typer.secho("\nNo matches found.", fg=typer.colors.YELLOW)
        return

    # Sort: EXACT -> DUPLICATE -> SIMILAR
    type_order = {"EXACT": 0, "DUPLICATE": 1, "SIMILAR": 2}
    matches.sort(key=lambda x: (type_order.get(x["type"], 3), x["dist"]))

    line_width = 140
    report_lines = [
        "=" * line_width,
        f"MATCH SEARCH REPORT ({datetime.datetime.now()})",
        f"Parameters: Duplicate < {dup_thresh}, Similar < {sim_thresh}",
        "=" * line_width,
        f"{'TYPE':<12} | {'DIST.':<8} | {'NEW FILE':<55} | {'ORIGINAL':<55}",
        "-" * line_width
    ]

    for m in matches:
        report_lines.append(
            f"{m['type']:<12} | {m['dist']:.4f} | {m['pretty_new']:<55} | {m['pretty_old']:<55}"
        )

    summary = f"\nTOTAL: {len(matches)} matches found (EXACT: {len([x for x in matches if x['type'] == 'EXACT'])})"
    report_lines.append(summary)
    full_report = "\n".join(report_lines)

    typer.echo("\n" + full_report)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_report)
    typer.secho(f"Report saved: {filename}", fg=typer.colors.GREEN)


def launch_visualization(matches: List[MatchResult], dataset_name: str):
    """Creates a FiftyOne dataset for the results and launches the app."""
    if not matches:
        return

    samples_to_add = []
    for idx, m in enumerate(matches):
        pair_id = f"match_{idx:05d}"

        # Add both images in the pair to the visualization dataset
        samples_to_add.append(fo.Sample(
            filepath=m['path_new'],
            pair_id=pair_id,
            role="NEW",
            match_type=m['type'],
            distance=m['dist']
        ))
        samples_to_add.append(fo.Sample(
            filepath=m['path_old'],
            pair_id=pair_id,
            role="OLD",
            match_type=m['type'],
            distance=m['dist']
        ))

    viz_dataset = fo.Dataset(dataset_name)
    viz_dataset.add_samples(samples_to_add)
    viz_dataset.compute_metadata()

    typer.secho("\nLaunching FiftyOne App...", fg=typer.colors.BRIGHT_YELLOW)
    session = fo.launch_app(viz_dataset.sort_by("pair_id"))
    input("\n>>> Press [ENTER] to exit...")


# --- Main Command ---

@app.command()
def find_duplicates(
        paths: List[str] = typer.Argument(...,
                                          help="Paths to folders separated by space (or a single string with ';')"),
        dup_thresh: float = typer.Option(0.04, "--dup", help="Threshold for duplicates (embeddings)"),
        sim_thresh: float = typer.Option(0.14, "--sim", help="Threshold for similar images (embeddings)"),
        viz: bool = typer.Option(True, "--viz/--no-viz", help="Whether to launch the FiftyOne web interface")
):
    """
    Search for exact duplicates (via hash) and visually similar images (via CLIP embeddings).
    """
    MODEL_NAME = "clip-vit-base32-torch"
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_FILE = f"report_{TIMESTAMP}.txt"
    VIZ_DATASET_NAME = f"comparison_{TIMESTAMP}"

    # 1. Setup
    dataset_paths = parse_input_paths(paths)
    typer.secho(f"Loading model {MODEL_NAME}...", fg=typer.colors.CYAN)
    model = foz.load_zoo_model(MODEL_NAME)

    loaded_datasets = []
    loaded_embeddings = []
    global_file_hashes: HashEntry = {}

    # 2. Process Datasets (Load, Hash, Embed)
    typer.secho("\nLoading datasets and indexing hashes...", fg=typer.colors.CYAN)
    for i, path in enumerate(dataset_paths):
        ds, emb, local_hashes = load_and_process_dataset(path, i + 1, model)
        if ds:
            loaded_datasets.append(ds)
            loaded_embeddings.append(emb)

            # Merge local hashes into global dict
            for h, paths in local_hashes.items():
                if h not in global_file_hashes:
                    global_file_hashes[h] = []
                global_file_hashes[h].extend(paths)

    # 3. Find Matches
    exact_matches, processed_pairs = find_exact_matches(global_file_hashes)

    visual_matches = find_visual_matches(
        loaded_datasets,
        loaded_embeddings,
        processed_pairs,
        dup_thresh,
        sim_thresh
    )

    all_matches = exact_matches + visual_matches

    # 4. Report and Visualize
    generate_text_report(all_matches, REPORT_FILE, dup_thresh, sim_thresh)

    if viz and all_matches:
        launch_visualization(all_matches, VIZ_DATASET_NAME)


if __name__ == "__main__":
    app()

