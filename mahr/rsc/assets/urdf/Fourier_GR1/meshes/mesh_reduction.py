#!/usr/bin/env python3
"""
Mesh Reduction Tool for STL files (Fourier_GR1 meshes)

This script reduces the number of faces (triangles) in STL meshes to be below a
specified maximum (default: 9,999) using Open3D's quadric decimation.

Usage examples:
  # From this directory (default processes current folder)
  python mesh_reduction.py

  # Explicit directory, keep reduced copies with suffix
  python mesh_reduction.py --dir ./ --max-faces 8000 --suffix _reduced

  # Overwrite originals in-place
  python mesh_reduction.py --overwrite

  # Dry-run: show what would be processed
  python mesh_reduction.py --dry-run

Notes:
  - Requires: open3d (pip install open3d)
  - Only STL files are processed (case-insensitive .stl/.STL).
  - Non-manifold/degenerate geometry is cleaned where possible before/after.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, TYPE_CHECKING, Any

# For type hints without importing Open3D at runtime
if TYPE_CHECKING:  # pragma: no cover - typing only
    from open3d.cpu.pybind.geometry import TriangleMesh  # type: ignore
else:
    TriangleMesh = Any  # fallback for runtime when open3d isn't imported

# Lazy import to allow --help without requiring open3d
try:
    import open3d as o3d  # type: ignore
except Exception as e:  # pragma: no cover - environment-dependent
    o3d = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_open3d():
    if o3d is None:
        print("ERROR: open3d is required for mesh reduction.", file=sys.stderr)
        print("Install with: pip install open3d", file=sys.stderr)
        if _IMPORT_ERROR:
            print(f"Details: {_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(2)


def find_stl_files(directory: Path, recursive: bool = True) -> List[Path]:
    patterns = ["*.stl", "*.STL"]
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(sorted(directory.rglob(pat)))
    else:
        for pat in patterns:
            files.extend(sorted(directory.glob(pat)))
    # Deduplicate in case-insensitive filesystems (safety)
    seen = set()
    uniq: List[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def clean_mesh(mesh: "TriangleMesh") -> "TriangleMesh":
    # In-place cleanup helpers
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    # Ensure normals exist for STL writing (requires triangle normals)
    if hasattr(mesh, "has_triangle_normals"):
        try:
            if not mesh.has_triangle_normals():
                mesh.compute_triangle_normals()
        except Exception:
            # Fallback: attempt compute regardless
            mesh.compute_triangle_normals()
    else:
        # Older Open3D: just compute
        try:
            mesh.compute_triangle_normals()
        except Exception:
            pass
    if hasattr(mesh, "has_vertex_normals") and not mesh.has_vertex_normals():
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
    return mesh


def simplify_to_max_faces(mesh: "TriangleMesh", max_faces: int) -> "TriangleMesh":
    """Simplify mesh so triangle count <= max_faces using quadric decimation.

    If the first attempt overshoots (rare), iteratively reduce target until the
    result meets the constraint or no further reduction is possible.
    """
    current_faces = len(mesh.triangles)
    if current_faces <= max_faces:
        return mesh

    target = max(4, int(max_faces))

    # Try up to a few iterations to ensure we get under the threshold
    for _ in range(5):
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        clean_mesh(simplified)
        faces = len(simplified.triangles)
        if faces <= max_faces:
            return simplified
        # If still above, adjust target downwards more aggressively for next try
        # Reduce by 10% extra or at least 1 triangle
        target = max(4, min(target - 1, int(target * 0.9)))
        mesh = simplified  # continue simplifying from the latest result

    # Final guard: if still above, force one more pass to half, then accept
    if len(mesh.triangles) > max_faces:
        target = max(4, int(max_faces * 0.8))
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        clean_mesh(simplified)
        return simplified

    return mesh


def process_file(
    path: Path,
    max_faces: int,
    overwrite: bool,
    suffix: str,
    dry_run: bool,
) -> None:
    # In dry-run mode, don't require Open3D; just report intended action
    if dry_run and (o3d is None):
        print(f"- Would process: {path}")
        return

    _require_open3d()

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or not mesh.has_triangles() or len(mesh.triangles) == 0:
        print(f"- Skip (no triangles): {path}")
        return

    clean_mesh(mesh)
    original_faces = len(mesh.triangles)

    if original_faces <= max_faces:
        print(f"- OK (already <= {max_faces} faces): {path} [{original_faces} faces]")
        return

    print(f"- Simplify: {path.name} [{original_faces} -> <= {max_faces} faces]")
    if dry_run:
        return

    simplified = simplify_to_max_faces(mesh, max_faces=max_faces)
    # Ensure triangle normals before writing
    try:
        simplified.compute_triangle_normals()
    except Exception:
        pass
    final_faces = len(simplified.triangles)

    # Prepare output path
    out_path = path if overwrite else path.with_name(path.stem + suffix + path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_triangle_mesh(
        str(out_path), simplified, write_ascii=False
    )
    if success:
        status = "OK" if final_faces <= max_faces else "WARN> threshold not met"
        print(f"  -> {status}: {out_path.name} [{final_faces} faces]")
    else:
        print(f"  -> ERROR: failed to write {out_path}")


def main(argv: Iterable[str] | None = None) -> int:
    default_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Reduce STL face counts to be <= max-faces using Open3D quadric decimation."
        )
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help="Directory containing STL files (default: this script's directory)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=9999,
        help="Maximum number of faces to keep (default: 9999)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original STL files in-place",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_reduced",
        help="Suffix for output files when not overwriting (default: _reduced)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not traverse subdirectories (default: recursive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without writing any files",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.max_faces < 4:
        print("ERROR: --max-faces must be >= 4", file=sys.stderr)
        return 2

    if not args.dir.exists() or not args.dir.is_dir():
        print(f"ERROR: directory not found or not a directory: {args.dir}", file=sys.stderr)
        return 2

    files = find_stl_files(args.dir, recursive=not args.no_recursive)
    if not files:
        print(f"No STL files found under: {args.dir}")
        return 0

    print(
        f"Found {len(files)} STL file(s) under {args.dir}. Target: <= {args.max_faces} faces."
    )
    if args.dry_run:
        print("Dry-run mode: no files will be modified.")

    for f in files:
        try:
            process_file(
                path=f,
                max_faces=args.max_faces,
                overwrite=bool(args.overwrite),
                suffix=str(args.suffix),
                dry_run=bool(args.dry_run),
            )
        except Exception as e:  # robust per-file handling
            print(f"- ERROR processing {f}: {e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
