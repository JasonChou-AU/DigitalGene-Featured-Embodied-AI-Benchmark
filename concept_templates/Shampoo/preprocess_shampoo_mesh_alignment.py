import argparse
import os
from typing import List, Tuple

import numpy as np


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-10:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180deg: choose an axis orthogonal to a.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        # Rodrigues for theta=pi: R = -I + 2 * axis * axis^T
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return r


def _load_obj_lines_and_vertices(path: str) -> Tuple[List[str], np.ndarray, List[int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    vertices = []
    vertex_line_ids = []
    for i, line in enumerate(lines):
        if line.startswith("v "):
            parts = line.strip().split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                vertex_line_ids.append(i)

    if len(vertices) == 0:
        raise ValueError(f"no vertices found in OBJ: {path}")
    return lines, np.asarray(vertices, dtype=np.float64), vertex_line_ids


def _write_obj_with_vertices(path: str, lines: List[str], vertex_line_ids: List[int], vertices: np.ndarray) -> None:
    out_lines = list(lines)
    for i, line_id in enumerate(vertex_line_ids):
        x, y, z = vertices[i]
        out_lines[line_id] = f"v {x:.9f} {y:.9f} {z:.9f}\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


def align_shampoo_mesh(input_obj: str, output_obj: str, scale: float = 0.01) -> None:
    lines, v, vertex_line_ids = _load_obj_lines_and_vertices(input_obj)

    # Principal axis from covariance; this approximates bottle/shampoo longitudinal direction.
    center = np.mean(v, axis=0)
    vc = v - center
    cov = (vc.T @ vc) / float(len(vc))
    w, vecs = np.linalg.eigh(cov)
    principal_axis = vecs[:, int(np.argmax(w))]
    principal_axis = principal_axis / (np.linalg.norm(principal_axis) + 1e-12)

    # Force principal axis to point to +Y hemisphere to avoid upside-down results.
    if principal_axis[1] < 0.0:
        principal_axis = -principal_axis

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    r = _rotation_from_a_to_b(principal_axis, y_axis)
    v_rot = (r @ v.T).T

    # Recenter to geometric center at origin.
    center_after_rot = np.mean(v_rot, axis=0)
    v_rot = v_rot - center_after_rot

    # Uniform scale to meter-like size.
    v_rot = v_rot * float(scale)

    _write_obj_with_vertices(output_obj, lines, vertex_line_ids, v_rot)

    principal_after = np.linalg.eigh(np.cov(v_rot.T))[1][:, -1]
    principal_after = principal_after / (np.linalg.norm(principal_after) + 1e-12)
    angle_to_y = np.degrees(np.arccos(np.clip(abs(float(np.dot(principal_after, y_axis))), -1.0, 1.0)))
    print(f"[OK] saved: {output_obj}")
    print(f"[INFO] principal axis angle to +Y after alignment: {angle_to_y:.4f} deg")
    print(f"[INFO] scale applied: {scale}")
    print(f"[INFO] mean center: {np.mean(v_rot, axis=0)}")
    print(f"[INFO] bbox min: {np.min(v_rot, axis=0)}")
    print(f"[INFO] bbox max: {np.max(v_rot, axis=0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Shampoo OBJ principal axis to +Y and recenter for Isaac.")
    parser.add_argument(
        "--input-obj",
        default="real_object_data/Shampoo/segmentation/Shampoo.obj",
        help="Input OBJ path.",
    )
    parser.add_argument(
        "--output-obj",
        default="real_object_data/Shampoo/segmentation/Shampoo_aligned.obj",
        help="Output OBJ path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Uniform scale multiplier applied after alignment.",
    )
    args = parser.parse_args()

    align_shampoo_mesh(os.path.abspath(args.input_obj), os.path.abspath(args.output_obj), scale=args.scale)


if __name__ == "__main__":
    main()
