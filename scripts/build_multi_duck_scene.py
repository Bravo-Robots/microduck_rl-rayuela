#!/usr/bin/env python3
"""
build_multi_duck_scene.py (v2)
================================

Genera una escena MJCF con N copias del Microduck a partir de tu
`scene.xml` real (el mismo que usa `scripts/infer_policy.py`). A
diferencia de la v1, esta versión resuelve `<include>` con
`mujoco.MjSpec` antes de duplicar, así que funciona directamente sobre
scene.xml aunque este incluya robot_walk.xml / meshes / etc.

No depende de ningún keyframe: la pose inicial (de pie, joints en
DEFAULT_POSE) la fija el controlador en runtime, igual que hace
infer_policy.py para 1 robot (ver multi_duck_controller.py).

Uso
---
    cd microduck_rl
    python build_multi_duck_scene.py \
        --scene src/mjlab_microduck/robot/microduck/scene.xml \
        --n 6 --layout line --spacing 0.5 \
        --out src/mjlab_microduck/robot/microduck/scene_line_6ducks.xml

    python build_multi_duck_scene.py \
        --scene src/mjlab_microduck/robot/microduck/scene.xml \
        --n 8 --layout race --cols 4 --spacing 0.5 \
        --out src/mjlab_microduck/robot/microduck/scene_race_8ducks.xml

    # pista de atletismo: un pato por carril, línea de salida + meta a
    # 6 m, piso terracota y carriles marcados en blanco
    python build_multi_duck_scene.py \
        --scene src/mjlab_microduck/robot/microduck/scene.xml \
        --n 6 --layout track --spacing 0.5 --track-length 6.0 \
        --out src/mjlab_microduck/robot/microduck/scene_track_6ducks.xml

IMPORTANTE: escribe el --out DENTRO de la misma carpeta que scene.xml
(src/mjlab_microduck/robot/microduck/). MjSpec.to_xml() conserva las
rutas de los assets (mallas STL, etc.) tal como estaban en el XML
original, normalmente relativas a esa carpeta — si guardas la escena
generada en otro sitio, MuJoCo no encontrará los .stl.
"""

import argparse
import copy
import xml.etree.ElementTree as ET

import mujoco

REF_ATTRS = {
    "name", "joint", "joint1", "joint2", "site", "site1", "site2",
    "body", "body1", "body2", "geom", "geom1", "geom2",
    "tendon", "objname", "sensor", "actuator",
}
NEVER_RENAME = {"mesh", "material", "texture", "class", "childclass", "hfield"}


def resolve_includes(scene_xml_path):
    """Usa MjSpec para resolver <include>/<compiler meshdir=...> y devuelve
    la raíz ElementTree del XML ya aplanado (un solo documento)."""
    spec = mujoco.MjSpec.from_file(scene_xml_path)
    flat_xml = spec.to_xml()
    return ET.fromstring(flat_xml)


def collect_local_names(root):
    names = set()
    for el in root.iter():
        if el.tag in NEVER_RENAME:
            continue
        n = el.get("name")
        if n:
            names.add(n)
    return names


def prefix_names(el, local_names, prefix):
    for attr in list(el.attrib.keys()):
        if attr in NEVER_RENAME:
            continue
        if attr in REF_ATTRS:
            val = el.get(attr)
            if val in local_names:
                el.set(attr, prefix + val)
    for child in el:
        prefix_names(child, local_names, prefix)


def find_robot_root_bodies(worldbody):
    return list(worldbody.findall("body"))


def make_placements(n, layout, spacing, cols, start_x):
    placements = []
    if layout in ("line", "track"):
        # en "track" cada pato queda centrado en su propio carril
        for i in range(n):
            dy = (i - (n - 1) / 2.0) * spacing
            placements.append((start_x, dy))
    elif layout == "race":
        cols = max(1, cols)
        row_spacing = spacing * 1.5
        for i in range(n):
            row = i // cols
            col = i % cols
            n_in_row = min(cols, n - row * cols)
            dy = (col - (n_in_row - 1) / 2.0) * spacing
            dx = start_x - row * row_spacing
            placements.append((dx, dy))
    else:
        raise ValueError(f"layout desconocido: {layout}")
    return placements


def build_stripe_line(x, y_half_width, n_stripes=12, z=0.001, width=0.04):
    """Línea a rayas blanco/negro perpendicular a la pista (para la meta)."""
    geoms = []
    stripe_w = (2 * y_half_width) / n_stripes
    for i in range(n_stripes):
        y0 = -y_half_width + i * stripe_w
        color = "1 1 1 1" if i % 2 == 0 else "0.05 0.05 0.05 1"
        geoms.append(ET.Element("geom", {
            "type": "box",
            "size": f"{width/2:.4f} {stripe_w/2:.4f} {z}",
            "pos": f"{x} {y0 + stripe_w/2:.4f} {z}",
            "rgba": color,
            "contype": "0", "conaffinity": "0",
        }))
    return geoms


def build_solid_line_across(x, y_half_width, z=0.001, width=0.04, rgba="1 1 1 1"):
    """Línea blanca sólida perpendicular a la pista (para la salida)."""
    return [ET.Element("geom", {
        "type": "box",
        "size": f"{width/2:.4f} {y_half_width:.4f} {z}",
        "pos": f"{x} 0 {z}",
        "rgba": rgba,
        "contype": "0", "conaffinity": "0",
    })]


def build_lane_lines(x_from, x_to, n_lanes, lane_width, z=0.001, width=0.03,
                      rgba="1 1 1 1"):
    """n_lanes+1 líneas blancas paralelas a la dirección de carrera (los
    carriles de una pista de atletismo), de x_from a x_to."""
    geoms = []
    x_center = (x_from + x_to) / 2.0
    length = abs(x_to - x_from)
    half_width = n_lanes * lane_width / 2.0
    for i in range(n_lanes + 1):
        y = -half_width + i * lane_width
        geoms.append(ET.Element("geom", {
            "type": "box",
            "size": f"{length/2:.4f} {width/2:.4f} {z}",
            "pos": f"{x_center:.4f} {y:.4f} {z}",
            "rgba": rgba,
            "contype": "0", "conaffinity": "0",
        }))
    return geoms


def build_scene(scene_xml_path, n, layout, spacing, cols, start_x, out_path,
                 track_length=6.0):
    flat_root = resolve_includes(scene_xml_path)

    worldbody = flat_root.find("worldbody")
    if worldbody is None:
        raise ValueError("El XML resuelto no tiene <worldbody>.")
    root_bodies = find_robot_root_bodies(worldbody)
    if len(root_bodies) != 1:
        names = [b.get("name") for b in root_bodies]
        raise ValueError(
            f"Se esperaba 1 body raíz de robot en <worldbody>, se "
            f"encontraron {len(root_bodies)}: {names}. Si scene.xml trae "
            f"más de un body de primer nivel (p.ej. una pelota), edita este "
            f"script para elegir el correcto por nombre."
        )
    template_body = root_bodies[0]
    local_names = collect_local_names(template_body)

    actuator_el = flat_root.find("actuator")
    sensor_el = flat_root.find("sensor")
    tendon_el = flat_root.find("tendon")
    equality_el = flat_root.find("equality")
    asset_el = flat_root.find("asset")
    default_el = flat_root.find("default")
    compiler_el = flat_root.find("compiler")

    # Los nombres locales relevantes para renombrar también incluyen los
    # definidos en actuator/sensor/tendon/equality (no solo el body tree).
    for section in (actuator_el, sensor_el, tendon_el, equality_el):
        if section is not None:
            local_names |= collect_local_names(section)

    placements = make_placements(n, layout, spacing, cols, start_x)

    out_root = ET.Element("mujoco", {"model": f"microduck_{layout}_{n}"})
    if compiler_el is not None:
        out_root.append(copy.deepcopy(compiler_el))
    if default_el is not None:
        out_root.append(copy.deepcopy(default_el))
    if asset_el is not None:
        out_root.append(copy.deepcopy(asset_el))
    # timestep/opciones físicas del original (si las hay)
    option_el = flat_root.find("option")
    if option_el is not None:
        out_root.append(copy.deepcopy(option_el))

    out_worldbody = ET.SubElement(out_root, "worldbody")
    ET.SubElement(out_worldbody, "light", {
        "directional": "true", "diffuse": "0.9 0.9 0.9",
        "pos": "0 0 5", "dir": "0 0 -1",
    })

    floor_rgba = "0.72 0.33 0.24 1" if layout == "track" else "0.3 0.5 0.3 1"
    ET.SubElement(out_worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05", "rgba": floor_rgba,
    })

    if layout == "race":
        for g in build_stripe_line(start_x, y_half_width=spacing * cols * 0.6):
            out_worldbody.append(g)

    if layout == "track":
        finish_x = start_x + track_length
        y_half_width = n * spacing / 2.0
        for g in build_lane_lines(start_x, finish_x, n_lanes=n, lane_width=spacing):
            out_worldbody.append(g)
        for g in build_solid_line_across(start_x, y_half_width):
            out_worldbody.append(g)
        for g in build_stripe_line(finish_x, y_half_width):
            out_worldbody.append(g)

    out_actuator = ET.SubElement(out_root, "actuator") if actuator_el is not None else None
    out_sensor = ET.SubElement(out_root, "sensor") if sensor_el is not None else None
    out_tendon = ET.SubElement(out_root, "tendon") if tendon_el is not None else None
    out_equality = ET.SubElement(out_root, "equality") if equality_el is not None else None

    for i, (dx, dy) in enumerate(placements):
        prefix = f"duck{i}_"

        body_copy = copy.deepcopy(template_body)
        prefix_names(body_copy, local_names, prefix)
        # Colocamos x,y; conservamos la z/orientación original del body
        # (la pose real de pie la fija el controlador en runtime).
        orig_pos = body_copy.get("pos", "0 0 0").split()
        z = orig_pos[2] if len(orig_pos) == 3 else "0"
        body_copy.set("pos", f"{dx:.4f} {dy:.4f} {z}")
        out_worldbody.append(body_copy)

        for src_section, dst_section in (
            (actuator_el, out_actuator),
            (sensor_el, out_sensor),
            (tendon_el, out_tendon),
            (equality_el, out_equality),
        ):
            if src_section is None:
                continue
            for child in src_section:
                child_copy = copy.deepcopy(child)
                prefix_names(child_copy, local_names, prefix)
                dst_section.append(child_copy)

    ET.indent(out_root, space="  ")
    ET.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Escena escrita en {out_path}  ({n} patos, layout={layout})")

    # sanity check: que MuJoCo la compile de verdad
    try:
        test_model = mujoco.MjModel.from_xml_path(out_path)
        print(f"  OK: compila en MuJoCo -> nq={test_model.nq} nu={test_model.nu} "
              f"nbody={test_model.nbody}")
    except Exception as e:
        print(f"  [AVISO] la escena se escribió pero MuJoCo no la pudo cargar: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                     help="tu scene.xml real (el mismo que usa infer_policy.py)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--layout", choices=["line", "race", "track"], default="line",
                     help="line: fila simple. race: parrilla por columnas. "
                          "track: pista de atletismo con carriles, línea de "
                          "salida y línea de meta (un pato por carril).")
    ap.add_argument("--spacing", type=float, default=0.5,
                     help="separación entre patos/carriles (m)")
    ap.add_argument("--cols", type=int, default=4, help="columnas en layout=race")
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--track-length", type=float, default=6.0,
                     help="distancia entre salida y meta en layout=track (m)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_scene(args.scene, args.n, args.layout, args.spacing, args.cols,
                args.start_x, args.out, track_length=args.track_length)


if __name__ == "__main__":
    main()
