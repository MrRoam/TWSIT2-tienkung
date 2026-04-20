#!/usr/bin/env python3
from __future__ import annotations

import copy
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SRC_XML = Path("/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_v1.xml")
DST_XML = Path("/data/shared_folder/GMR/assets/tienkung_ei/mjcf/tienkung_ei_custom_collision.xml")
MESH_DIR = SRC_XML.parent.parent / "meshes"


KEEP_COLLISION_BODIES = {
    "waist_yaw_link": ("cylinder", 1),
    "hip_yaw_l_link": ("capsule", 1),
    "knee_pitch_l_link": ("capsule", 1),
    "ankle_roll_l_link": ("capsule", 2),
    "hip_yaw_r_link": ("capsule", 1),
    "knee_pitch_r_link": ("capsule", 1),
    "ankle_roll_r_link": ("capsule", 2),
    "shoulder_roll_l_link": ("capsule", 1),
    "shoulder_yaw_l_link": ("capsule", 1),
    "elbow_pitch_l_link": ("capsule", 1),
    "shoulder_roll_r_link": ("capsule", 1),
    "shoulder_yaw_r_link": ("capsule", 1),
    "elbow_pitch_r_link": ("capsule", 1),
}

# Match the G1 custom-collision spirit more closely for the torso:
# use one manually sized, centered vertical cylinder instead of a raw mesh-AABB fit.
MANUAL_COLLISION_OVERRIDES = {
    "waist_yaw_link": {
        "type": "cylinder",
        "pos": np.array([0.0, 0.0, 0.23], dtype=np.float64),
        "size": np.array([0.09, 0.18], dtype=np.float64),
    },
    # Match the G1 training asset's foot treatment: two slim longitudinal
    # cylinders per foot instead of broad auto-fit capsules.
    "ankle_roll_l_link": {
        "type": "cylinder_pair_x",
        "centers": [
            np.array([0.05, 0.02, -0.02], dtype=np.float64),
            np.array([0.05, -0.02, -0.02], dtype=np.float64),
        ],
        "radius": 0.02,
        "half_length": 0.075,
    },
    "ankle_roll_r_link": {
        "type": "cylinder_pair_x",
        "centers": [
            np.array([0.05, 0.02, -0.02], dtype=np.float64),
            np.array([0.05, -0.02, -0.02], dtype=np.float64),
        ],
        "radius": 0.02,
        "half_length": 0.075,
    },
}


def load_stl_vertices(stl_path: Path) -> np.ndarray:
    raw = stl_path.read_bytes()
    if raw[:5].lower() == b"solid":
        try:
            vertices = []
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("vertex"):
                    _, x, y, z = line.split()
                    vertices.append((float(x), float(y), float(z)))
            if vertices:
                return np.asarray(vertices, dtype=np.float64)
        except Exception:
            pass

    if len(raw) < 84:
        raise ValueError(f"STL too small: {stl_path}")
    tri_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + tri_count * 50
    if len(raw) < expected:
        raise ValueError(f"Binary STL truncated: {stl_path}")

    data = np.frombuffer(raw, dtype=np.uint8, offset=84, count=tri_count * 50)
    data = data.reshape(tri_count, 50)
    verts = np.empty((tri_count * 3, 3), dtype=np.float64)
    for i in range(tri_count):
        verts_block = np.frombuffer(data[i, 12:48].tobytes(), dtype="<f4").reshape(3, 3)
        verts[i * 3 : (i + 1) * 3] = verts_block
    return verts


def mesh_bounds(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = MESH_DIR / f"{mesh_name}.STL"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh for {mesh_name}: {mesh_path}")
    vertices = load_stl_vertices(mesh_path)
    return vertices.min(axis=0), vertices.max(axis=0)


def primary_mesh_name(body: ET.Element) -> str | None:
    body_name = body.attrib.get("name")
    for geom in body.findall("geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name == body_name:
            return mesh_name
    for geom in body.findall("geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            return mesh_name
    return None


def visual_only(geom: ET.Element) -> None:
    geom.attrib["contype"] = "0"
    geom.attrib["conaffinity"] = "0"
    geom.attrib["group"] = "2"
    if "class" not in geom.attrib:
        geom.attrib["class"] = "visual"


def add_box_collision(body: ET.Element, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
    extents = (bbox_max - bbox_min) * 0.9
    center = (bbox_min + bbox_max) * 0.5
    geom = ET.SubElement(body, "geom")
    geom.attrib.update(
        {
            "class": "collision",
            "type": "box",
            "group": "3",
            "size": " ".join(f"{v:.6f}" for v in extents * 0.5),
            "pos": " ".join(f"{v:.6f}" for v in center),
        }
    )


def add_cylinder_collision(body: ET.Element, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
    extents = bbox_max - bbox_min
    axis = int(np.argmax(extents))
    ordered = np.argsort(extents)
    longest = float(extents[axis])
    radius_extent = float(extents[ordered[0]])
    center = (bbox_min + bbox_max) * 0.5

    # MuJoCo cylinders are aligned with the local z-axis by default.
    quat = None
    if axis == 0:
        quat = "0.707107 0.000000 0.707107 0.000000"
    elif axis == 1:
        quat = "0.707107 -0.707107 0.000000 0.000000"

    geom = ET.SubElement(body, "geom")
    geom.attrib.update(
        {
            "class": "collision",
            "type": "cylinder",
            "group": "3",
            "size": f"{max(1e-4, 0.45 * radius_extent):.6f} {0.45 * longest:.6f}",
            "pos": " ".join(f"{v:.6f}" for v in center),
        }
    )
    if quat is not None:
        geom.attrib["quat"] = quat


def add_manual_cylinder_collision(body: ET.Element, pos: np.ndarray, size: np.ndarray) -> None:
    geom = ET.SubElement(body, "geom")
    geom.attrib.update(
        {
            "class": "collision",
            "type": "cylinder",
            "group": "3",
            "size": f"{float(size[0]):.6f} {float(size[1]):.6f}",
            "pos": " ".join(f"{float(v):.6f}" for v in pos),
        }
    )


def add_manual_cylinder_pair_x_collision(
    body: ET.Element, centers: list[np.ndarray], radius: float, half_length: float
) -> None:
    quat = "0.707107 0.000000 0.707107 0.000000"
    for center in centers:
        geom = ET.SubElement(body, "geom")
        geom.attrib.update(
            {
                "class": "collision",
                "type": "cylinder",
                "group": "3",
                "size": f"{float(radius):.6f} {float(half_length):.6f}",
                "pos": " ".join(f"{float(v):.6f}" for v in center),
                "quat": quat,
            }
        )


def axis_vector(idx: int) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float64)
    vec[idx] = 1.0
    return vec


def add_capsule_collision(body: ET.Element, bbox_min: np.ndarray, bbox_max: np.ndarray, count: int) -> None:
    extents = bbox_max - bbox_min
    axis = int(np.argmax(extents))
    ordered = np.argsort(extents)
    second = extents[ordered[-2]]
    longest = extents[axis]
    shortest_nonzero = max(float(extents[ordered[0]]), 1e-4)
    center = (bbox_min + bbox_max) * 0.5
    axis_dir = axis_vector(axis)

    if count == 1:
        half_length = 0.45 * float(longest)
        radius = max(1e-4, 0.30 * float(second))
        start = center - axis_dir * half_length
        end = center + axis_dir * half_length
        geom = ET.SubElement(body, "geom")
        geom.attrib.update(
            {
                "class": "collision",
                "type": "capsule",
                "group": "3",
                "size": f"{radius:.6f}",
                "fromto": " ".join(f"{v:.6f}" for v in np.concatenate([start, end])),
            }
        )
        return

    radius = max(1e-4, 0.35 * float(shortest_nonzero))
    half_length = 0.18 * float(longest)
    offsets = (-0.18 * float(longest), 0.22 * float(longest))
    for offset in offsets:
        capsule_center = center + axis_dir * offset
        start = capsule_center - axis_dir * half_length
        end = capsule_center + axis_dir * half_length
        geom = ET.SubElement(body, "geom")
        geom.attrib.update(
            {
                "class": "collision",
                "type": "capsule",
                "group": "3",
                "size": f"{radius:.6f}",
                "fromto": " ".join(f"{v:.6f}" for v in np.concatenate([start, end])),
            }
        )


def main() -> None:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Missing worldbody in source XML")

    for body in worldbody.iter("body"):
        for geom in body.findall("geom"):
            visual_only(geom)

        body_name = body.attrib.get("name")
        if body_name not in KEEP_COLLISION_BODIES:
            continue

        mesh_name = primary_mesh_name(body)
        if mesh_name is None:
            raise RuntimeError(f"No mesh geom found for body {body_name}")

        bbox_min, bbox_max = mesh_bounds(mesh_name)
        geom_type, geom_count = KEEP_COLLISION_BODIES[body_name]
        manual_override = MANUAL_COLLISION_OVERRIDES.get(body_name)
        if manual_override is not None:
            if manual_override["type"] == "cylinder":
                add_manual_cylinder_collision(body, manual_override["pos"], manual_override["size"])
            elif manual_override["type"] == "cylinder_pair_x":
                add_manual_cylinder_pair_x_collision(
                    body,
                    manual_override["centers"],
                    manual_override["radius"],
                    manual_override["half_length"],
                )
            continue
        if geom_type == "box":
            add_box_collision(body, bbox_min, bbox_max)
        elif geom_type == "cylinder":
            add_cylinder_collision(body, bbox_min, bbox_max)
        else:
            add_capsule_collision(body, bbox_min, bbox_max, geom_count)

    tree.write(DST_XML, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {DST_XML}")


if __name__ == "__main__":
    main()
