"""
How the 3-D views turn, and how specimens are lined up, run under Node.

The camera turns all the way round (it once stopped a degree short of the
top), rolls, and a specimen turned or moved alone follows the pointer as the
camera would. Each specimen turns about its tissue's centre, and the Compare
page fits every specimen to the first by the rigid move that makes their tissue
lie closest (web/js/pose.js): these tests hold it to known turns of a shaped
phantom, in both masked and unmasked previews.

Run with:  python -m pytest studio/tests/test_pose.py
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

WEB = Path(__file__).resolve().parents[1] / "web" / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")

TURN = 0.008  # radians per pixel dragged (render3d.js)
FOV = math.radians(30)


def js(tmp_path: Path, body: str, **inputs):
    """The JSON value `body` returns, with pose.js as P, render3d.js as R, and each input bound."""
    data = tmp_path / "inputs.json"
    data.write_text(json.dumps(inputs))
    script = tmp_path / "run.mjs"
    script.write_text(
        f"import * as P from {json.dumps((WEB / 'pose.js').as_uri())};\n"
        f"import * as R from {json.dumps((WEB / 'render3d.js').as_uri())};\n"
        "import { readFileSync } from 'node:fs';\n"
        f"const I = JSON.parse(readFileSync({json.dumps(str(data))}, 'utf8'));\n"
        "const canvas = () => ({ getContext: () => null, clientWidth: 400, clientHeight: 300 });\n"
        "// A preview from the tissue's voxel indices: masked (0 outside), or over a noisy background.\n"
        "const volume = (v) => {\n"
        "  const data = new Uint16Array(v.shape[0] * v.shape[1] * v.shape[2]);\n"
        "  if (v.background) data.set(v.background);\n"
        "  for (const i of v.tissue) data[i] = 40000;\n"
        "  const quant = v.masked ? { lo: 0, hi: 65534, offset: 1, scale: 65534 } : { lo: 0, hi: 65535, offset: 0, scale: 65535 };\n"
        "  return { data, meta: { shape: v.shape, spacing_preview_mm: v.spacing, masked: v.masked, quant, histogram: v.histogram } };\n"
        "};\n"
        f"const out = (() => {{ {body} }})();\n"
        "console.log(JSON.stringify(out));\n")
    done = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def rotation(axis, angle):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def turn_between(a, b):
    """Degrees between two rotations."""
    m = np.asarray(a).reshape(3, 3) @ np.asarray(b).reshape(3, 3).T
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(m) - 1) / 2))))


# A hemimandible, roughly, in mm about its own origin: a long body, a ramus
# rising at the back to a condyle, three molar crowns on one side of the top
# and an incisor tip at the front -- lopsided on every axis, as a real one is.
BLOBS = [((0.0, 0.0, 0.0), (0.45, 0.10, 0.08)), ((-0.30, 0.18, 0.0), (0.10, 0.18, 0.05)),
         ((-0.35, 0.36, 0.02), (0.05, 0.05, 0.05)), ((-0.05, 0.12, 0.04), (0.04, 0.04, 0.04)),
         ((0.05, 0.12, 0.04), (0.04, 0.04, 0.04)), ((0.15, 0.12, 0.04), (0.04, 0.04, 0.04)),
         ((0.45, -0.05, 0.0), (0.04, 0.04, 0.04))]


def world(shape, spacing):
    """Every voxel's centre in the renderer's world frame (x along axis 0, y against axis 1, z along axis 2)."""
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    sz, sy, sx = spacing
    return np.stack([(z + 0.5) * sz - shape[0] * sz / 2, shape[1] * sy / 2 - (y + 0.5) * sy,
                     (x + 0.5) * sx - shape[2] * sx / 2], axis=-1)


def phantom(shape, spacing, turn=np.eye(3), at=(0.0, 0.0, 0.0), blobs=BLOBS):
    """Tissue voxel indices of the shape turned by `turn` and set at `at` (mm from the box's centre)."""
    p = (world(shape, spacing) - np.asarray(at)) @ turn  # each voxel back in the shape's own frame
    inside = np.zeros(shape, bool)
    for centre, semi in blobs:
        inside |= (((p - centre) / semi) ** 2).sum(-1) <= 1
    return inside


def masked(inside, spacing):
    return {"shape": list(inside.shape), "spacing": list(spacing), "masked": True,
            "tissue": np.flatnonzero(inside).tolist()}


SPACING = (0.025, 0.025, 0.025)


# ---------------------------------------------------------------- the camera

def test_the_camera_turns_all_the_way_over_the_top_and_back(tmp_path):
    step = math.pi / 90 / TURN  # 2 degrees of drag
    got = js(tmp_path, """
      const r = new R.Render3D(canvas());
      const start = r.camera(), halves = [];
      for (let k = 1; k <= 180; k++) { r.rotate(0, I.step); if (k === 90) halves.push(r.camera()); }
      const round = r.camera();
      for (let k = 0; k < 180; k++) r.rotate(I.step, 0);
      return { start, half: halves[0], round, sideways: r.camera() };
    """, step=step)
    start, half = got["start"], got["half"]
    # Half a turn down carries the eye over the top to the far side, upside down:
    # the old camera stopped a degree short of the top.
    assert half["eye"] == pytest.approx([-v for v in start["eye"]], abs=1e-6)
    assert half["up"] == pytest.approx([-v for v in start["up"]], abs=1e-6)
    for whole in (got["round"], got["sideways"]):
        for k in ("eye", "up", "right"):
            assert whole[k] == pytest.approx(start[k], abs=1e-6)


def test_a_roll_turns_the_picture_anticlockwise_about_the_view(tmp_path):
    got = js(tmp_path, "const r = new R.Render3D(canvas()); const a = r.camera(); r.roll(0.3); return [a, r.camera()];")
    before, after = got
    assert after["forward"] == pytest.approx(before["forward"], abs=1e-9)
    # The world's old "up" now points up and to the left on screen.
    up = np.array(before["up"])
    assert [float(up @ after["right"]), float(up @ after["up"])] == pytest.approx([-math.sin(0.3), math.cos(0.3)], abs=1e-9)


def test_turning_one_specimen_alone_looks_as_turning_the_camera_would(tmp_path):
    # The specimen turned by a drag must look exactly as the camera turned by it,
    # or a linked view would drift from the one lined up alone.
    got = js(tmp_path, """
      const a = new R.Render3D(canvas()), b = new R.Render3D(canvas());
      a.setView('end'); b.setView('end');
      for (const [dx, dy] of [[40, 25], [-15, 60], [5, -80]]) { a.turnModel(dx, dy); b.rotate(dx, dy); }
      a.rollModel(0.4); b.roll(0.4);
      return [a.camera().local, b.camera().local];
    """)
    alone, camera = got
    for k in ("eye", "forward", "right", "up"):
        assert alone[k] == pytest.approx(camera[k], abs=1e-9)


def test_moving_one_specimen_alone_follows_the_pointer(tmp_path):
    got = js(tmp_path, """
      const r = new R.Render3D(canvas());
      r.radius = 1; r.setView('side', 6);
      r.rotate(30, -20);
      r.moveModel(120, -40);
      const f = r.camera();
      return { shift: r.pose.shift, eye: f.eye, forward: f.forward, right: f.right, up: f.up, fit: r.fitDistance() };
    """)
    rel = np.subtract(got["shift"], got["eye"])
    depth = float(rel @ got["forward"])
    px = 150 / (depth * math.tan(FOV / 2))  # the canvas is 300 pixels high
    assert [float(rel @ got["right"]) * px, -float(rel @ got["up"]) * px] == pytest.approx([120, -40], abs=1e-6)
    assert depth == pytest.approx(6, abs=1e-9)  # across the view, not towards it
    # The view that fits it reaches as far as it was moved.
    assert got["fit"] == pytest.approx((1 + np.linalg.norm(got["shift"])) / math.sin(FOV / 2) * 1.02)


# ---------------------------------------------------------------- the tissue

def test_principal_axes_match_numpy(tmp_path):
    rng = np.random.default_rng(4)
    mats = []
    for _ in range(6):
        a = rng.normal(size=(3, 3))
        mats.append((a @ a.T).ravel().tolist())
    mats.append([2, 0, 0, 0, 2, 0, 0, 0, 1])  # a repeated value
    got = js(tmp_path, "return I.mats.map((m) => P.eigenSym(m));", mats=mats)
    for m, e in zip(mats, got):
        m = np.reshape(m, (3, 3))
        values, vectors = np.array(e["values"]), np.reshape(e["vectors"], (3, 3))
        assert values == pytest.approx(sorted(np.linalg.eigvalsh(m), reverse=True), abs=1e-9)
        assert vectors @ np.diag(values) @ vectors.T == pytest.approx(m, abs=1e-9)
        assert np.linalg.det(vectors) == pytest.approx(1.0)


def test_a_specimen_turns_about_its_tissue_not_its_box(tmp_path):
    inside = phantom((48, 44, 46), SPACING, at=(0.08, -0.05, 0.03))
    rng = np.random.default_rng(1)
    # The same specimen unmasked, in air at 2,000-4,000: its Otsu cut finds it.
    background = rng.integers(2000, 4000, inside.size)
    raw = np.where(inside.ravel(), 40000, background)
    counts, edges = np.histogram(raw, bins=128, range=(0, 65535))
    unmasked = {"shape": list(inside.shape), "spacing": list(SPACING), "masked": False, "tissue": np.flatnonzero(inside).tolist(),
                "background": background.tolist(), "histogram": {"counts": counts.tolist(), "edges": edges.tolist()}}
    got = js(tmp_path, "return [I.a, I.b].map((v) => { const c = P.tissueCloud(volume(v)); return { centre: c.centre, radius: c.radius, count: c.count }; });",
             a=masked(inside, SPACING), b=unmasked)
    points = world(inside.shape, SPACING)[inside]
    centroid = points.mean(0)
    reach = np.linalg.norm(points - centroid, axis=1).max()
    for cloud in got:
        assert cloud["count"] == inside.sum()
        assert cloud["centre"] == pytest.approx(centroid.tolist(), abs=1e-6)
        assert reach <= cloud["radius"] <= reach + 0.05


# ---------------------------------------------------------------- lining up

@pytest.mark.parametrize("axis, degrees, at", [
    ((0.3, 0.8, 0.5), 140, (0.05, -0.04, 0.02)),   # turned right round in its box
    ((1, 0, 0), 25, (-0.06, 0.03, 0.05)),          # rolled about its own long axis
    ((0.2, 1, -0.4), 180, (0.0, 0.0, 0.0)),        # upside down, end to end
    ((0, 0, 1), 12, (0.1, 0.0, -0.05)),            # tipped a little, as in a holder
])
def test_a_turned_specimen_is_fitted_back_onto_the_first(tmp_path, axis, degrees, at):
    first = phantom((48, 44, 46), SPACING)
    turn = rotation(axis, math.radians(degrees))
    second = phantom((56, 56, 56), SPACING, turn=turn, at=at)
    got, itself = js(tmp_path, "const a = P.tissueCloud(volume(I.a));"
                     "return [P.register(a, P.tissueCloud(volume(I.b))), P.register(a, a)];",
                     a=masked(first, SPACING), b=masked(second, SPACING))
    # A point of the second, about its centre, lands where the first has it: the turn undone,
    # their surfaces a fraction of a voxel apart, as the first's is from itself.
    assert turn_between(got["rot"], turn.T) < 3
    assert np.linalg.norm(got["shift"]) < SPACING[0]
    assert itself["misfit"] < 0.1 * SPACING[0]
    assert got["misfit"] < 0.5 * SPACING[0]


def test_one_already_in_line_is_left_as_scanned(tmp_path):
    first = phantom((48, 44, 46), SPACING)
    second = phantom((56, 44, 50), SPACING, at=(0.07, -0.02, 0.06))
    # An ellipsoid looks the same flipped end to end: a flip would fit as well, but is not taken.
    egg = [((0.0, 0.0, 0.0), (0.4, 0.15, 0.1))]
    got = js(tmp_path, "return [[I.a, I.b], [I.c, I.d]].map(([x, y]) => P.register(P.tissueCloud(volume(x)), P.tissueCloud(volume(y))));",
             a=masked(first, SPACING), b=masked(second, SPACING),
             c=masked(phantom((40, 30, 30), SPACING, blobs=egg), SPACING),
             d=masked(phantom((44, 36, 30), SPACING, at=(0.03, 0.02, 0.0), blobs=egg), SPACING))
    for fit in got:
        assert math.degrees(fit["turned"]) < 2


def test_one_missing_part_of_its_tissue_is_still_fitted_the_right_way_round(tmp_path):
    # Its ramus cropped off, a hemimandible is nearly symmetric end to end: the
    # molars and the incisor tip alone tell it from its flip, which fits 3% worse.
    first = phantom((48, 44, 46), SPACING)
    no_ramus = [BLOBS[0]] + BLOBS[3:]
    turns = [rotation((0.3, 0.8, 0.5), math.radians(140)), rotation((0.2, 1, -0.4), math.pi)]
    inputs = {f"b{i}": masked(phantom((56, 56, 56), SPACING, turn=t, at=(0.03, 0.02, -0.02), blobs=no_ramus), SPACING)
              for i, t in enumerate(turns)}
    got = js(tmp_path, "const a = P.tissueCloud(volume(I.a));"
             "return [I.b0, I.b1].map((b) => P.register(a, P.tissueCloud(volume(b))));", a=masked(first, SPACING), **inputs)
    for fit, turn in zip(got, turns):
        assert turn_between(fit["rot"], turn.T) < 4


def test_the_distance_to_a_surface_is_exact(tmp_path):
    ndimage = pytest.importorskip("scipy.ndimage")
    seeds = (np.random.default_rng(0).random((23, 17, 31)) < 0.004).astype(np.uint8)
    got = js(tmp_path, "return Array.from(P.distanceField(Uint8Array.from(I.seeds), I.dims, 0.5));",
             seeds=seeds.ravel().tolist(), dims=list(seeds.shape))
    assert np.reshape(got, seeds.shape) == pytest.approx(ndimage.distance_transform_edt(1 - seeds) * 0.5, abs=1e-5)
