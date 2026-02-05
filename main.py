import hashlib
import os
import datetime
import numpy as np
import typer
import fiftyone as fo
import fiftyone.zoo as foz
from typing import List, Optional, Dict
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

app = typer.Typer(rich_markup_mode="rich")


def get_file_hash(path: str) -> str:
    """Calculates the MD5 hash of a file to find exact duplicates."""
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception:
        return ""
    return hash_md5.hexdigest()


def format_path(full_path: str) -> str:
    """Formats the path to show: folder1/folder2/filename.jpg"""
    parts = Path(full_path).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else full_path


@app.command()
def find_duplicates(
        paths: List[str] = typer.Argument(..., help="Paths to folders separated by space (or a single string with ';')"),
        dup_thresh: float = typer.Option(0.04, "--dup", help="Threshold for duplicates (embeddings)"),
        sim_thresh: float = typer.Option(0.14, "--sim", help="Threshold for similar images (embeddings)"),
        viz: bool = typer.Option(True, "--viz/--no-viz", help="Whether to launch the FiftyOne web interface")
):
    """
    Search for exact duplicates (via hash) and visually similar images (via CLIP embeddings).
    """
    # Parse paths (supporting both spaces and ';')
    if len(paths) == 1 and ";" in paths[0]:
        dataset_paths = [p.strip() for p in paths[0].split(";") if p.strip()]
    else:
        dataset_paths = paths

    MODEL_NAME = "clip-vit-base32-torch"
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_FILE = f"report_{TIMESTAMP}.txt"
    VIZ_DATASET_NAME = f"comparison_{TIMESTAMP}"

    typer.secho(f"Loading model {MODEL_NAME}...", fg=typer.colors.CYAN)
    model = foz.load_zoo_model(MODEL_NAME)

    all_file_hashes: Dict[str, List[str]] = {}
    loaded_datasets = []
    loaded_embeddings = []
    found_matches = []
    processed_exact_pairs = set()  # To avoid duplicating hash-matches in embedding search

    # 1. Loading and Hashing
    typer.secho("\nLoading datasets and indexing hashes...", fg=typer.colors.CYAN)
    for i, path in enumerate(dataset_paths):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            typer.secho(f"ERROR: Path does not exist: {abs_path}", fg=typer.colors.RED)
            continue

        folder_name = os.path.basename(abs_path.rstrip(os.sep))
        ds_name = f"source_{i + 1}_{folder_name}"

        if ds_name in fo.list_datasets():
            fo.delete_dataset(ds_name)

        ds = fo.Dataset.from_dir(dataset_dir=abs_path, dataset_type=fo.types.ImageDirectory, name=ds_name)
        ds.persistent = True

        # Calculate hashes for all files in this set
        for sample in ds:
            h = get_file_hash(sample.filepath)
            if h:
                if h not in all_file_hashes:
                    all_file_hashes[h] = []
                all_file_hashes[h].append(sample.filepath)

        typer.echo(f" [{ds_name}] images: {len(ds)}. Computing embeddings...")
        emb = ds.compute_embeddings(model=model)

        loaded_datasets.append(ds)
        loaded_embeddings.append(emb)

    # 2. Search for exact duplicates via HASH (EXACT)
    typer.secho("\nSearching for exact matches via hash...", fg=typer.colors.GREEN, bold=True)
    for h, file_list in all_file_hashes.items():
        if len(file_list) > 1:
            # Compare all files within the same hash group
            original = file_list[0]
            for duplicate in file_list[1:]:
                found_matches.append({
                    "type": "EXACT",
                    "dist": 0.0,
                    "pretty_new": format_path(duplicate),
                    "pretty_old": format_path(original),
                    "path_new": duplicate,
                    "path_old": original
                })
                # Store this pair to prevent adding it again via embeddings
                pair_key = tuple(sorted([duplicate, original]))
                processed_exact_pairs.add(pair_key)

    # 3. Search for visual matches (DUPLICATE/SIMILAR)
    typer.secho("Searching for visual matches (CLIP)...", fg=typer.colors.MAGENTA, bold=True)
    for i in range(1, len(loaded_datasets)):
        query_ds = loaded_datasets[i]
        query_emb_raw = loaded_embeddings[i]
        query_paths = query_ds.values("filepath")

        valid_idxs = [idx for idx, e in enumerate(query_emb_raw) if e is not None]
        if not valid_idxs: continue

        query_emb = np.array([query_emb_raw[idx] for idx in valid_idxs])
        query_paths_v = [query_paths[idx] for idx in valid_idxs]

        for j in range(i):
            index_ds = loaded_datasets[j]
            index_emb_raw = loaded_embeddings[j]
            index_paths = index_ds.values("filepath")

            v_idx_index = [idx for idx, e in enumerate(index_emb_raw) if e is not None]
            if not v_idx_index: continue

            index_emb = np.array([index_emb_raw[idx] for idx in v_idx_index])
            index_paths_v = [index_paths[idx] for idx in v_idx_index]

            typer.echo(f"  Comparing: {query_ds.name} vs {index_ds.name}")

            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(index_emb)
            distances, indices = nn.kneighbors(query_emb)

            for k in range(len(query_emb)):
                dist = float(distances[k][0])
                if dist <= sim_thresh:
                    path_new = query_paths_v[k]
                    path_old = index_paths_v[indices[k][0]]

                    # Skip if this pair was already found via hash
                    if tuple(sorted([path_new, path_old])) in processed_exact_pairs:
                        continue

                    m_type = "DUPLICATE" if dist <= dup_thresh else "SIMILAR"
                    found_matches.append({
                        "type": m_type,
                        "dist": dist,
                        "pretty_new": format_path(path_new),
                        "pretty_old": format_path(path_old),
                        "path_new": path_new,
                        "path_old": path_old
                    })

    # 4. Report Generation and FiftyOne integration
    if found_matches:
        # Sort order: EXACT -> DUPLICATE -> SIMILAR
        type_order = {"EXACT": 0, "DUPLICATE": 1, "SIMILAR": 2}
        found_matches.sort(key=lambda x: (type_order.get(x["type"], 3), x["dist"]))

        line_width = 140
        report_lines = [
            "=" * line_width,
            f"MATCH SEARCH REPORT ({TIMESTAMP})",
            f"Parameters: Duplicate < {dup_thresh}, Similar < {sim_thresh}",
            "=" * line_width,
            f"{'TYPE':<12} | {'DIST.':<8} | {'NEW FILE (folder/folder/file)':<55} | {'ORIGINAL':<55}",
            "-" * line_width
        ]

        samples_to_add = []
        for idx, m in enumerate(found_matches):
            report_lines.append(f"{m['type']:<12} | {m['dist']:.4f} | {m['pretty_new']:<55} | {m['pretty_old']:<55}")

            pair_id = f"match_{idx:05d}"
            samples_to_add.append(fo.Sample(filepath=m['path_new'], pair_id=pair_id, role="NEW", match_type=m['type'],
                                            distance=m['dist']))
            samples_to_add.append(fo.Sample(filepath=m['path_old'], pair_id=pair_id, role="OLD", match_type=m['type'],
                                            distance=m['dist']))

        summary = f"\nTOTAL: {len(found_matches)} matches found (EXACT: {len([x for x in found_matches if x['type'] == 'EXACT'])})"
        report_lines.append(summary)

        full_report = "\n".join(report_lines)
        typer.echo("\n" + full_report)

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(full_report)
        typer.secho(f"Report saved: {REPORT_FILE}", fg=typer.colors.GREEN)

        if viz:
            viz_dataset = fo.Dataset(VIZ_DATASET_NAME)
            viz_dataset.add_samples(samples_to_add)
            viz_dataset.compute_metadata()
            typer.secho("\nLaunching FiftyOne App...", fg=typer.colors.BRIGHT_YELLOW)
            session = fo.launch_app(viz_dataset.sort_by("pair_id"))
            input("\n>>> Press [ENTER] to exit...")
    else:
        typer.secho("\nNo matches found.", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
