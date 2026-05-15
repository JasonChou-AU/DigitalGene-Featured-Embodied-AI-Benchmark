#!/usr/bin/env python3
"""Scale and center OBJ vertex positions.

Only ``v`` records are transformed. All other OBJ content is copied through so
faces, normals, texture coordinates, materials, comments, and object/group names
stay intact.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple


Vertex = Tuple[float, float, float]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALE = 100.0
CENTER_MODE = "bbox"

OBJ_PAIRS = [
    (
        Path("real_object_data/Mug/segmentation/Mug.obj"),
        Path("real_object_data/Mug/segmentation/Mug_aligned.obj"),
    ),
    (
        Path("real_object_data/Shampoo/segmentation/Cylindrical_Body.obj"),
        Path("real_object_data/Shampoo/segmentation/Cylindrical_Body_aligned.obj"),
    ),
    (
        Path("real_object_data/Shampoo/segmentation/Regular_nozzle_0.obj"),
        Path("real_object_data/Shampoo/segmentation/Regular_nozzle_0_aligned.obj"),
    ),
    (
        Path("real_object_data/Shampoo/segmentation/Regular_nozzle_1.obj"),
        Path("real_object_data/Shampoo/segmentation/Regular_nozzle_1_aligned.obj"),
    ),
    (
        Path("real_object_data/Shampoo/segmentation/shampoo.obj"),
        Path("real_object_data/Shampoo/segmentation/shampoo_aligned.obj"),
    ),
]


def parse_vertex(line: str) -> Optional[Vertex]:
    """Return the xyz coordinates from a vertex line, or None otherwise."""
    parts = line.strip().split()
    if not parts or parts[0] != "v" or len(parts) < 4:
        return None

    try:
        return float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None


def load_vertices(obj_path: Path) -> List[Vertex]:
    vertices: List[Vertex] = []
    with obj_path.open("r", encoding="utf-8", errors="replace") as obj_file:
        for line in obj_file:
            vertex = parse_vertex(line)
            if vertex is not None:
                vertices.append(vertex)
    return vertices


def compute_center(vertices: Sequence[Vertex], center_mode: str) -> Vertex:
    if not vertices:
        raise ValueError("OBJ file does not contain any vertex records")

    if center_mode == "centroid":
        count = float(len(vertices))
        return (
            sum(v[0] for v in vertices) / count,
            sum(v[1] for v in vertices) / count,
            sum(v[2] for v in vertices) / count,
        )

    if center_mode == "bbox":
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        return (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            (min(zs) + max(zs)) / 2.0,
        )

    raise ValueError(f"unsupported center mode: {center_mode}")


def transform_vertex(vertex: Vertex, scale: float, center: Vertex) -> Vertex:
    return (
        vertex[0] / scale - center[0],
        vertex[1] / scale - center[1],
        vertex[2] / scale - center[2],
    )


def format_vertex_line(original_line: str, transformed: Vertex) -> str:
    parts = original_line.rstrip("\r\n").split()
    suffix = ""
    if original_line.endswith("\r\n"):
        suffix = "\r\n"
    elif original_line.endswith("\n"):
        suffix = "\n"

    # Preserve optional OBJ vertex attributes after xyz, such as vertex color.
    extra = " ".join(parts[4:])
    line = f"v {transformed[0]:.8f} {transformed[1]:.8f} {transformed[2]:.8f}"
    if extra:
        line = f"{line} {extra}"
    return line + suffix


def align_obj(
    input_path: Path,
    output_path: Path,
    *,
    scale: float,
    center_mode: str,
) -> int:
    if scale == 0:
        raise ValueError("scale must not be zero")

    original_vertices = load_vertices(input_path)
    scaled_vertices = [transform_vertex(vertex, scale, (0.0, 0.0, 0.0)) for vertex in original_vertices]
    center = compute_center(scaled_vertices, center_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_index = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line in src:
            vertex = parse_vertex(line)
            if vertex is None:
                dst.write(line)
                continue

            transformed = transform_vertex(vertex, scale, center)
            dst.write(format_vertex_line(line, transformed))
            vertex_index += 1

    return vertex_index


def main() -> int:
    for input_obj, output_obj in OBJ_PAIRS:
        input_path = PROJECT_ROOT / input_obj
        output_path = PROJECT_ROOT / output_obj
        count = align_obj(
            input_path,
            output_path,
            scale=SCALE,
            center_mode=CENTER_MODE,
        )
        print(f"Wrote {output_obj} ({count} vertices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
