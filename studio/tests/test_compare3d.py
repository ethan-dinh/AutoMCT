"""
The compare page's one heatmap scale, run under Node.

The page shows several specimens in 3-D with one colour window, set in mg
HA/cm³ and turned back into each specimen's stored values with its own HA
calibration. These tests run web/js/scale.js and the 3-D renderer's own
texture mapping under Node and hold them to the promise the page makes: one
density gets one colour in every panel, whatever each scan's calibration,
scaling or preview range -- including a window that reaches past a preview's
own values, which the renderer once clamped, stretching that specimen's colours.
They also pin the shared window and how the panels are laid out.

Run with:  python -m pytest studio/tests/test_compare3d.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web" / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")


def js(expression: str, **inputs):
    """The JSON value of `expression`, with scale.js as S, render3d.js as R, colormaps.js as C and each input bound."""
    bindings = "".join(f"const {k} = {json.dumps(v)};\n" for k, v in inputs.items())
    code = (f"import * as S from {json.dumps((WEB / 'scale.js').as_uri())};\n"
            f"import * as R from {json.dumps((WEB / 'render3d.js').as_uri())};\n"
            f"import * as C from {json.dumps((WEB / 'colormaps.js').as_uri())};\n"
            f"{bindings}console.log(JSON.stringify({expression}));")
    done = subprocess.run([NODE, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# Three scans on three footings: a Scanco ISQ's calibration at mu_scaling 4096;
# the same scanner's at 2048 with a shifted noise floor; another scanner's,
# on a conversion rescaled to 1/cm. Each preview spans its own stored range.
SPECIMENS = [
    {"cal": {"slope": 375.507, "intercept": -197.288, "divisor": 4096.0, "offset": 0.0},
     "quant": {"lo": 4000.0, "hi": 32000.0}, "masked": True},
    {"cal": {"slope": 375.507, "intercept": -197.288, "divisor": 2048.0, "offset": 12.5},
     "quant": {"lo": 1500.0, "hi": 16000.0}, "masked": True},
    {"cal": {"slope": 410.0, "intercept": -230.0, "divisor": 1.0, "offset": 0.0},
     "quant": {"lo": 2.0, "hi": 8.5}, "masked": False},
]

# Where a density falls in the window, in each specimen, as the renderer's
# shader computes it: the stored value through the texture's mapping (norm),
# against the window's two ends through the same mapping.
POSITIONS = """windows.map(([lo, hi]) => densities.map((d) => specimens.map(({cal, quant, masked}) => {
  const norm = (v) => R.Render3D.prototype.norm.call({ quant: { q: quant, masked } }, v);
  const [a, b] = [norm(S.fromScale(lo, cal)), norm(S.fromScale(hi, cal))];
  return Math.min(1, Math.max(0, (norm(S.fromScale(d, cal)) - a) / (b - a)));
})))"""


def test_one_density_gets_one_colour_whatever_the_calibration():
    windows = [[900.0, 2400.0], [1500.0, 2100.0]]
    densities = [1000.0, 1500.0, 1930.0, 2300.0]
    got = js(POSITIONS, specimens=SPECIMENS, windows=windows, densities=densities)
    for (lo, hi), rows in zip(windows, got):
        for d, per_specimen in zip(densities, rows):
            want = min(1.0, max(0.0, (d - lo) / (hi - lo)))
            assert per_specimen == pytest.approx([want] * len(SPECIMENS), abs=1e-9)


def test_a_window_beyond_a_previews_own_values_keeps_its_width():
    # -500 and 3200 mg HA/cm³ lie outside every preview's stored range; clamped
    # to it, each specimen's colours would stretch by its own range instead.
    windows = [[-500.0, 3200.0]]
    densities = [0.0, 1180.0, 1930.0]
    got = js(POSITIONS, specimens=SPECIMENS, windows=windows, densities=densities)
    for d, per_specimen in zip(densities, got[0]):
        assert per_specimen == pytest.approx([(d + 500.0) / 3700.0] * len(SPECIMENS), abs=1e-9)


def test_the_scale_converts_both_ways_and_raw_values_pass_through():
    got = js("specimens.map(({cal}) => [S.toScale(S.fromScale(1930, cal), cal), S.fromScale(S.toScale(23200, cal), cal)])"
             ".concat([[S.toScale(23200, null), S.fromScale(23200, null)]])", specimens=SPECIMENS)
    for there_and_back in got[:-1]:
        assert there_and_back == pytest.approx([1930.0, 23200.0])
    assert got[-1] == [23200, 23200]
    # The ISQ's own calibration: water (0.4741/cm, stored x4096) near 0, enamel near 1930.
    cal = SPECIMENS[0]["cal"]
    assert js("[S.toScale(0.4741 * 4096, cal), S.toScale(23200, cal)]", cal=cal) == pytest.approx(
        [-19.3, 1929.6], abs=0.1)


def test_the_window_spans_every_specimen_on_the_shared_scale():
    edges = [float(10 * i) for i in range(11)]
    specs = [
        # Stored as mg HA/cm³; tissue in 30-40 and 80-90.
        {"hist": {"edges": edges, "counts": [0, 0, 0, 5, 0, 0, 0, 0, 5, 0]}, "masked": True,
         "cal": {"slope": 1.0, "intercept": 0.0, "divisor": 1.0, "offset": 0.0}},
        # mg = value / 2 + 10; tissue in 20-30 and 90-100: 22.5 to 57.5 on the shared scale.
        {"hist": {"edges": edges, "counts": [0, 0, 5, 0, 0, 0, 0, 0, 0, 5]}, "masked": True,
         "cal": {"slope": 2.0, "intercept": 10.0, "divisor": 4.0, "offset": 0.0}},
    ]
    assert js("S.sharedWindow(specs)", specs=specs) == {"lo": pytest.approx(22.5), "hi": pytest.approx(85.0)}
    # An unmasked scan's air peak is not tissue: the window starts above it.
    air = [{"hist": {"edges": edges, "counts": [900, 0, 4, 0, 0, 0, 0, 3, 0, 0]}, "masked": False, "cal": None}]
    assert js("S.sharedWindow(specs)", specs=air) == {"lo": pytest.approx(25.0), "hi": pytest.approx(75.0)}
    assert js("S.sharedWindow([])") is None


@pytest.mark.parametrize("n, width, height, cols", [
    (1, 1100, 700, 1), (2, 1100, 700, 2), (3, 1100, 700, 2), (4, 1100, 700, 2),
    (4, 2400, 600, 4), (2, 500, 900, 1), (3, 1800, 500, 3),
])
def test_panels_are_laid_out_for_the_most_room(n, width, height, cols):
    assert js(f"S.gridColumns({n}, {width}, {height})") == cols


def test_the_texture_holds_each_value_as_its_nearest_half_float():
    import numpy as np
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.uniform(0, 1, 4000), rng.uniform(0, 0.01, 500), [0.0, 1.0, 1 / 255, 0.5, 2.0 ** -20]])
    bits = js("values.map((v) => R.toHalf(v))", values=values.tolist())
    got = np.array(bits, dtype=np.uint16).view(np.float16).astype(np.float64)
    nearest = values.astype(np.float16).astype(np.float64)
    assert np.all(np.abs(got - values) <= np.abs(nearest - values) + 1e-12)
    # Eight times finer than the 8-bit texture it replaced, where enamel sits (0.5-1).
    top = values[values >= 0.5]
    assert np.max(np.abs(got[values >= 0.5] - top)) <= 2.0 ** -12


def test_steps_fall_on_the_same_densities_in_every_specimen():
    # Every 25 mg HA/cm³ from zero: each specimen's texture puts 1,950 on an edge and 1,962.5 mid-step.
    got = js("""specimens.map(({cal, quant, masked}) => {
      const r = Object.create(R.Render3D.prototype);
      r.quant = { q: quant, masked };
      const s = r.stepUniforms(S.stepsFor({ origin: 0, width: 25, fill: true, lines: true }, cal));
      return densities.map((d) => (r.norm(S.fromScale(d, cal)) - s.origin) / s.width);
    })""", specimens=SPECIMENS, densities=[1950.0, 1962.5, 2000.0, 2011.0])
    for bands in got:
        assert bands == pytest.approx([78.0, 78.5, 80.0, 80.44], abs=1e-6)
    assert js("Object.assign(Object.create(R.Render3D.prototype), {quant: {q: {lo: 0, hi: 1}, masked: false}}).stepUniforms(null)") == {
        "origin": 0, "width": 0, "fill": False, "lines": False}


def test_the_legend_colours_a_step_by_its_middle_as_the_panels_do():
    scale = {"lo": 1800.0, "hi": 2100.0, "gamma": 1.0}
    stepped = {**scale, "steps": {"origin": 0, "width": 50, "fill": True}}
    got = js("[S.rampPosition(1930, scale), S.rampPosition(1930, stepped), S.rampPosition(1900, stepped),"
             " S.rampPosition(1700, stepped), S.rampPosition(1930, {...scale, gamma: 2})]", scale=scale, stepped=stepped)
    assert got == pytest.approx([130 / 300, 125 / 300, 125 / 300, 0.0, (130 / 300) ** 0.5])


def test_steps_are_round_and_about_six_across_the_window():
    assert js("[S.stepWidth(1800, 2100), S.stepWidth(1100, 1960), S.stepWidth(0, 1), S.stepWidth(19000, 26000)]") == \
        pytest.approx([50, 100, 0.2, 1000])
    # Fine steps too: every width from about 3 to about 200 steps across the window.
    assert js("S.stepChoices(1800, 2100)") == [2, 2.5, 5, 10, 20, 25, 50, 100]
    assert js("[S.stepWidth(1663, 2354, 48), S.stepWidth(1663, 2354, 96)]") == [10, 5]


def test_rainbow_steps_each_get_their_own_colour_blue_to_red():
    # 48 steps of 10 across 1,700-2,180: every step a colour of its own, the lowest blue and the highest red.
    got = js("""(() => {
      const scale = { lo: 1700, hi: 2180, gamma: 1, steps: { origin: 0, width: 10, fill: true } };
      return Array.from({ length: 48 }, (_, k) => C.colorAt("rainbow", S.rampPosition(1705 + 10 * k, scale)));
    })()""")
    assert len({tuple(c) for c in got}) == 48
    (r0, g0, b0), (r1, g1, b1) = got[0], got[-1]
    assert b0 > r0 and r1 > b1
    # Nowhere near black, so no step disappears into the dark stage.
    assert min(max(c) for c in got) > 150


def test_random_steps_give_each_step_in_the_window_a_colour_unlike_its_neighbours():
    # Steps of 50 over 1,663-2,354: the 15 from 1,650 to 2,400, each a colour of its own however they are dealt.
    got = js("""[0, 7, 123456].map((seed) => {
      const steps = S.randomSteps(1663, 2354, 50, seed), scale = { lo: 1663, hi: 2354, steps };
      const colours = Array.from({ length: 15 }, (_, k) => S.stepColour(Math.min(2354, 1675 + 50 * k), scale));
      return { origin: steps.origin, colours, gaps: colours.slice(1).map((c, k) => C.unlike(c, colours[k])),
               outside: [S.stepColour(1662.9, scale), S.stepColour(2354.1, scale)],
               grey: Math.min(...colours.map((c) => Math.min(C.unlike(c, C.OUTSIDE.below), C.unlike(c, C.OUTSIDE.above)))) };
    })""")
    for deal in got:
        assert deal["origin"] == 1650
        assert len({tuple(c) for c in deal["colours"]}) == 15
        # Neighbours never alike, and no step mistaken for the grey outside the window.
        assert min(deal["gaps"]) > 0.1 and deal["grey"] > 0.1
        assert deal["outside"] == [[96, 104, 106], [222, 227, 226]]
    assert len({str(deal["colours"]) for deal in got}) == 3
    # Past MOST_COLOURS steps the colours come round again: of the 139 steps of 5, those 32 apart share one.
    cycle = js("""(() => {
      const scale = { lo: 1663, hi: 2354, steps: S.randomSteps(1663, 2354, 5) };
      return { n: scale.steps.palette.length, a: S.stepColour(1702, scale), b: S.stepColour(1702 + 32 * 5, scale),
               next: S.stepColour(1707, scale) };
    })()""")
    assert cycle["n"] == 32 and cycle["a"] == cycle["b"] != cycle["next"]


def test_random_steps_colour_the_same_densities_alike_in_every_specimen():
    # The renderer's steps, and the shader's k mod n, give each density the colour the legend does.
    got = js("""specimens.map(({cal, quant, masked}) => {
      const r = Object.create(R.Render3D.prototype);
      r.quant = { q: quant, masked };
      const steps = S.randomSteps(1663, 2354, 50, 3), scale = { lo: 1663, hi: 2354, steps };
      const s = r.stepUniforms(S.stepsFor(steps, cal)), n = s.palette.length;
      return densities.map((d) => {
        const k = Math.floor((r.norm(S.fromScale(d, cal)) - s.origin) / s.width);
        return [s.palette[k - n * Math.floor((k + 0.5) / n)], S.stepColour(d, scale)];
      });
    })""", specimens=SPECIMENS, densities=[1663.5, 1700.5, 1949.0, 2011.0, 2353.5])
    for specimen in got:
        for shader, legend in specimen:
            assert shader == legend
    assert {str(s) for s in got} == {str(got[0])}
