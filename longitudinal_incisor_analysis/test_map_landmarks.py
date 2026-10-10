"""
Tests for `map_landmarks.py`'s napari picker.

The viewer is stubbed, but the *layers* are real napari layer objects. That
distinction is the point of this file: a stub that merely accepts any kwargs
silently swallows layer-construction errors, and one of those shipped -- an
empty Shapes layer defaults to 2-D and rejects a 3-element `scale`, which only
surfaced when the tool was run on a real specimen. Constructing the real layers
here reproduces that class of failure without needing a GL context.

Run:  python test_map_landmarks.py
"""
import sys, numpy as np, tempfile, os, types
sys.path.insert(0,".")
import map_landmarks as M
from map_landmarks import (CMPRTransform, DEFAULT_LANDMARKS,
                           map_landmarks, write_csv, read_marks_csv)
from napari.layers import Image, Points, Shapes
from napari.components.layerlist import LayerList

d=tempfile.mkdtemp()
cl=np.stack([np.arange(200.),np.full(200,60.),np.full(200,60.)],axis=1)
np.savez(os.path.join(d,"S.npz"),centerline=cl,frames=np.zeros((200,2,3)),
  bbox_min=np.zeros(3),slice_size=np.int64(64),spacing=np.float64(1.),
  volume_shape=np.array([200,120,120]),voxel_spacing=np.array([0.008]*3),
  interpolation_order=np.int64(1))
tr=CMPRTransform(os.path.join(d,"S.npz"))

class StubDims:
    def __init__(s): s.current_step=[100,0,0]; s.ndisplay=2
    def set_current_step(s,ax,v): s.current_step[ax]=v
class StubViewer:
    def __init__(s,**k):
        s.dims=StubDims(); s.status=""; s.keymap={}
        s.scale_bar=types.SimpleNamespace(visible=False,unit="")
        # A real LayerList, so `.selection.active` behaves as in napari rather
        # than quietly accepting whatever a plain list would.
        s.layers=LayerList()
    def add_image(s,data,**k):
        l=Image(data,**k); s.layers.append(l); return l      # REAL layer
    def add_points(s,data,**k):
        l=Points(data,**k); s.layers.append(l); return l     # REAL layer
    def add_shapes(s,data=None,**k):
        l=Shapes(data,**k) if data is not None else Shapes(**k)
        s.layers.append(l); return l                          # REAL layer
    def bind_key(s,key,overwrite=False):
        def deco(fn): s.keymap[key]=fn; return fn
        return deco

captured={}
import napari as _napari
def _mkviewer(**k):
    v=StubViewer(**k); captured["v"]=v; return v
def _run():
    v=captured["v"]; km=v.keymap
    print("bound keys:",sorted(km))
    from map_landmarks import LANDMARK_KEYS, UNDO_KEY, TOGGLE_3D_KEY
    for k in [x[0] for x in LANDMARK_KEYS]+[UNDO_KEY,TOGGLE_3D_KEY]:
        assert k in km, f"missing {k}"
    # eruption: one key, one z
    v.dims.set_current_step(0,20); km["q"](v)
    # M1: separate start and end keys -- no press-count state
    v.dims.set_current_step(0,60);  km["g"](v)
    v.dims.set_current_step(0,110); km["h"](v)
    # pressing a key again MOVES its mark rather than advancing a state machine
    v.dims.set_current_step(0,70);  km["g"](v)
    # M2 start only -- half-marked, must survive as a point
    v.dims.set_current_step(0,150); km["j"](v)

    pl=[l for l in v.layers if l.name=="landmark planes"][0]
    assert pl.ndim==3 and pl.data[0].shape==(4,3), (pl.ndim,pl.data[0].shape)
    # eruption + M1 start + M1 end + M2 start = 4 planes (the move did not add one)
    assert len(pl.data)==4, len(pl.data)
    labels=sorted(str(x) for x in pl.features["label"])
    print("planes:",len(pl.data),"| labels:",labels)
    assert labels==["M1 end","M1 start","M2 start","eruption"], labels
    assert pl.mode=="pan_zoom" and not pl.editable

    km[UNDO_KEY](v)   # undo the M2 start
    assert len(pl.data)==3, len(pl.data)
    km[TOGGLE_3D_KEY](v); assert v.dims.ndisplay==3

# Patch only these two attributes, leaving napari importable for its own
# internals (the layer classes import napari.components at construction).
_napari.Viewer=_mkviewer
_napari.run=_run

vol=np.random.default_rng(0).integers(0,200,(200,120,120)).astype(np.int16)
ov=[("molar",np.zeros((200,120,120),np.int16),"magenta")]
out=M.pick_landmarks(vol,(0.008,)*3,tr,"Odam",(25,231),overlays=ov)
print("marks:",out)
got={m["landmark"]:(m["z_start"],m["z_end"]) for m in out}
assert got["eruption"]==(20,None), got
assert got["M1"]==(70,110), got       # start was MOVED from 60 to 70
assert "M2" not in got, got           # undone
print("OK  separate start/end keys, no press-count state")
print("OK  re-pressing a key moves its mark instead of advancing state")
print("OK  undo removed the M2 start")
print("OK  planes appear automatically, undo removes them")
print("OK  plane layer is read-only (pan_zoom, not editable)")
print("OK  real napari layers constructed (Image, Points, Shapes + overlay)")
print("\nREAL-LAYER PASS")


def test_moving_marks_and_undo_order():
    """Re-pressing a key moves its mark; undo follows press order across keys."""
    def run2():
        v=captured["v"]; km=v.keymap
        from map_landmarks import UNDO_KEY
        v.dims.set_current_step(0,10);  km["g"](v)          # M1 start
        v.dims.set_current_step(0,40);  km["h"](v)    # M1 end
        v.dims.set_current_step(0,70);  km["g"](v)          # move M1 start
        v.dims.set_current_step(0,90);  km["h"](v)    # move M1 end
        v.dims.set_current_step(0,120); km["q"](v)          # eruption
        v.dims.set_current_step(0,160); km["j"](v)          # M2 start
        km[UNDO_KEY](v)                                           # undo M2, not eruption
    _napari.run=run2
    out=M.pick_landmarks(np.zeros((200,120,120),np.int16),(0.008,)*3,tr,"t",(0,200),overlays=[])
    got={m["landmark"]:(m["z_start"],m["z_end"]) for m in out}
    print("after moves+undo:",got)
    assert got["M1"]==(70,90), got["M1"]        # both ends moved, still one span
    assert got["eruption"]==(120,None)
    assert "M2" not in got
    print("OK  re-pressing a key moves that mark, leaving the pair intact")
    print("OK  undo removed M2, leaving the earlier eruption mark intact")
    print("MOVE PASS")

test_moving_marks_and_undo_order()


def test_keys_do_not_collide_with_napari():
    """
    Every key this module binds must be free in napari.

    This is the check the tool shipped without. napari binds most single letters
    and the digits 1-4 to layer-mode switches at the viewer level, and its
    binding wins while the image layer is selected -- so a collision is silent:
    the key switches the image to pan/zoom instead of marking a landmark, and
    nothing raises. `1` and `2` were exactly this.
    """
    from map_landmarks import (assert_keys_available, napari_reserved_keys,
                               LANDMARK_KEYS, UNDO_KEY, TOGGLE_3D_KEY)
    reserved = napari_reserved_keys()
    assert reserved, "expected to read napari's shortcut table"
    problems = assert_keys_available()
    assert not problems, problems
    print(f"OK  all {len(LANDMARK_KEYS)+2} keys free of napari's "
          f"{len(reserved)} shortcuts")

    # The guard must actually fire on a known-bad key, or it proves nothing.
    import map_landmarks as _M
    saved = _M.LANDMARK_KEYS
    try:
        _M.LANDMARK_KEYS = (("1", "M1", "start"),)   # napari image pan/zoom
        bad = _M.assert_keys_available()
        assert bad and "already a napari shortcut" in bad[0], bad
        print("OK  guard catches '1' (napari:activate_image_pan_zoom_mode)")
        _M.LANDMARK_KEYS = (("!", "M1", "start"), ("@", "M1", "end"))
        bad2 = _M.assert_keys_available()
        assert any("normalises" in b for b in bad2), bad2
        print("OK  guard catches '!' and '@' collapsing to the same key")
    finally:
        _M.LANDMARK_KEYS = saved
    print("KEYS PASS")

test_keys_do_not_collide_with_napari()


# ---------------------------------------------------------------------------
# Projection math -- no viewer involved
# ---------------------------------------------------------------------------

def test_spans_and_events():
    """Span/event mapping, orientation ordering, and CSV round-trip."""
    d=tempfile.mkdtemp()
    # Straight centerline along z, 0..399, so z maps cleanly to slice index.
    cl=np.stack([np.arange(400.), np.full(400,50.), np.full(400,50.)],axis=1)
    np.savez(os.path.join(d,"S_cmpr_transform.npz"), centerline=cl, frames=np.zeros((400,2,3)),
      bbox_min=np.zeros(3), slice_size=np.int64(64), spacing=np.float64(1.),
      volume_shape=np.array([400,100,100]), voxel_spacing=np.array([0.01]*3),
      interpolation_order=np.int64(1))
    tr=CMPRTransform(os.path.join(d,"S_cmpr_transform.npz"))

    # 1. project_z picks the right slice
    p=tr.project_z(250)
    assert p["slice_index"]==250 and p["z_residual_vox"]==0
    print(f"OK  project_z(250) -> slice {p['slice_index']}, residual {p['z_residual_vox']}")

    # 2. event vs span
    marks=[{"landmark":"eruption","z_start":100,"z_end":None},
           {"landmark":"M1","z_start":200,"z_end":260}]
    rows=map_landmarks(tr,marks,"tip-high-z",tip_at_index_0=False)
    ev=[r for r in rows if r["kind"]=="event"][0]
    sp=[r for r in rows if r["kind"]=="span"][0]
    assert ev["z_percent_end"]=="" and ev["length_mm"]==""
    print(f"OK  event: {ev['landmark']} at {ev['z_percent_start']}%  (no end/length)")
    print(f"OK  span:  {sp['landmark']} {sp['z_percent_start']}% -> {sp['z_percent_end']}%, "
          f"len {sp['length_mm']}mm, extent {sp['extent_percent']}%")
    assert abs(sp["length_mm"]-0.60)<1e-6, sp["length_mm"]  # 60 vox * 0.01mm

    # 3. THE CRITICAL CASE: the orientation flag must NOT move z_percent.
    #
    # This assertion is the reverse of what it used to be. It previously
    # required the tip-low-z result to be the MIRROR of the tip-high-z one,
    # which baked in the bug: z_percent was derived by pushing a centerline
    # index through z_to_percent, flipping an axis that was already in
    # cervical -> tip order. See test_tip_low_z_not_mirrored below, which uses
    # a fixture where index order and volume-z order genuinely disagree (they
    # coincide here, since this centerline has z == index).
    rows_lo=map_landmarks(tr,[{"landmark":"M1","z_start":200,"z_end":260}],"tip-low-z",tip_at_index_0=False)[0]
    print(f"OK  tip-low-z span: {rows_lo['z_percent_start']}% -> {rows_lo['z_percent_end']}%")
    assert rows_lo["z_percent_start"] < rows_lo["z_percent_end"], "span not reordered!"
    assert abs(rows_lo["length_mm"]-0.60)<1e-6
    assert abs(rows_lo["z_percent_start"] - sp["z_percent_start"]) < 1e-6, \
        "orientation flag changed z_percent; it is provenance only"
    assert abs(rows_lo["z_percent_end"] - sp["z_percent_end"]) < 1e-6
    print("OK  orientation flag leaves z_percent unchanged, still start<end")

    # 4. marks given end-before-start still normalize
    rev=map_landmarks(tr,[{"landmark":"M2","z_start":300,"z_end":240}],"tip-high-z",tip_at_index_0=False)[0]
    assert rev["z_percent_start"]<rev["z_percent_end"]
    print(f"OK  reversed pick normalized: {rev['z_percent_start']}% -> {rev['z_percent_end']}%")

    # 5. sorted by position
    multi=map_landmarks(tr,[{"landmark":"M3","z_start":350,"z_end":380},
                            {"landmark":"M1","z_start":100,"z_end":150},
                            {"landmark":"eruption","z_start":50,"z_end":None}],"tip-high-z",tip_at_index_0=False)
    assert [r["landmark"] for r in multi]==["eruption","M1","M3"]
    print("OK  sorted by trajectory:",[r["landmark"] for r in multi])

    # 6. CSV round-trip
    out=os.path.join(d,"lm.csv"); write_csv(multi,out)
    print("\n"+open(out).read())
    back=read_marks_csv(out)
    again=map_landmarks(tr,back,"tip-high-z",tip_at_index_0=False)
    assert [r["z_percent_start"] for r in multi]==[r["z_percent_start"] for r in again]
    assert [r["z_percent_end"] for r in multi]==[r["z_percent_end"] for r in again]
    print("OK  CSV round-trips (spans and events both)")
    print("\nSPAN PASS")


def test_curved_centerline():
    """Arc length vs volume z on a curve, and out-of-range residuals."""
    d=tempfile.mkdtemp()
    # Curved: z advances then hooks back (like a tip curving over).
    t=np.linspace(0,1,400)
    z=300*t - 60*t**2*(t>0.7)   # slows near the end
    cl=np.stack([z, 50+40*np.sin(t*2), np.full(400,50.)],axis=1)
    np.savez(os.path.join(d,"C_cmpr_transform.npz"), centerline=cl, frames=np.zeros((400,2,3)),
      bbox_min=np.zeros(3), slice_size=np.int64(64), spacing=np.float64(1.),
      volume_shape=np.array([320,120,100]), voxel_spacing=np.array([0.01]*3),
      interpolation_order=np.int64(1))
    tr=CMPRTransform(os.path.join(d,"C_cmpr_transform.npz"))
    lo,hi=tr.z_span()
    print(f"centerline z span: {lo:.1f} .. {hi:.1f}")

    # arc length > z extent, because the tooth curves
    print(f"arc length {tr.arc_length[-1]:.1f} vox vs z extent {hi-lo:.1f} vox")
    assert tr.arc_length[-1] > (hi-lo), "curved centerline should be longer than its z extent"
    print("OK  arc length exceeds z extent (curvature accounted for)")

    # equal z steps -> unequal arc steps
    a=tr.project_z(50); b=tr.project_z(100); c=tr.project_z(150)
    d1=b["arc_length_mm"]-a["arc_length_mm"]; d2=c["arc_length_mm"]-b["arc_length_mm"]
    print(f"OK  equal z steps give unequal arc: {d1:.4f}mm vs {d2:.4f}mm")
    assert abs(d1-d2)>1e-6

    # a z beyond the centerline clamps, with a visible residual
    far=tr.project_z(hi+40)
    print(f"OK  z beyond centerline -> residual {far['z_residual_vox']:.1f} vox (flagged)")
    assert far["z_residual_vox"]>30
    print("\nCURVE PASS")


def test_centerline_traced_from_tip_not_mirrored():
    """
    A centerline traced tip -> cervical loop must not come out mirrored.

    Regression test for the real bug. `cmpr.py` does not promise which end it
    starts tracing from, and on the P82 specimens it starts at the TIP. The
    conversion nonetheless assumed index 0 was the cervical loop, so every
    landmark came out mirrored: molars ordered M3, M2, M1 with the eruption
    plane cervical to all of them, which is anatomically impossible.

    The fixture below reproduces that: index 0 is the tip, which is also the
    densest end, so `tip_at_index_0` must detect it and flip the percentages.
    """
    d = tempfile.mkdtemp()
    n = 400
    depth = 420
    cl = np.stack([np.linspace(400., 0., n),
                   np.full(n, 60.), np.full(n, 60.)], axis=1)
    np.savez(os.path.join(d, "T_cmpr_transform.npz"), centerline=cl,
             frames=np.zeros((n, 2, 3)), bbox_min=np.zeros(3),
             slice_size=np.int64(64), spacing=np.float64(1.),
             volume_shape=np.array([depth, 120, 120]),
             voxel_spacing=np.array([0.008] * 3),
             interpolation_order=np.int64(1))
    tr = CMPRTransform(os.path.join(d, "T_cmpr_transform.npz"))

    # A volume whose HIGH end of axis 0 is dense (mature enamel at the tip,
    # which the centerline reaches at index 0) and whose low end is soft.
    vol = np.zeros((depth, 120, 120), dtype=np.uint16)
    vol[350:, 50:70, 50:70] = 240          # tip: dense enamel
    vol[:60,  40:80, 40:80] = 90           # cervical loop: soft, wider

    assert tr.tip_at_index_0(vol) is True, \
        "tip should be detected at centerline index 0 (the dense end)"
    print("OK  tip detected at index 0 from mineralization")

    # The centerline runs tip (index 0, volume z=400) -> cervical loop (index
    # n-1, volume z=0), so on THIS volume tipward means HIGHER z. Marks are
    # written accordingly: eruption nearest the tip, then M1, M2, M3 receding
    # toward the cervical loop.
    marks = [
        {"landmark": "eruption", "z_start": 330., "z_end": None},
        {"landmark": "M1", "z_start": 260., "z_end": 300.},
        {"landmark": "M2", "z_start": 210., "z_end": 250.},
        {"landmark": "M3", "z_start": 160., "z_end": 200.},
    ]

    for orient in ("tip-low-z", "tip-high-z"):
        rows = map_landmarks(tr, marks, orientation=orient, volume=vol)
        pct = {r["landmark"]: r["z_percent_mid"] for r in rows}
        print(f"  [{orient}] " + "  ".join(
            f"{k}={pct[k]:.1f}%" for k in ("M1", "M2", "M3", "eruption")))

        # M1 is the mesial molar, nearest the incisor, so it sits TIPWARD of
        # M2 and M3 on this axis: M3 < M2 < M1. Eruption is tipward of all.
        assert pct["M3"] < pct["M2"] < pct["M1"], \
            f"molars mirrored under {orient}: {pct}"
        assert pct["eruption"] > pct["M1"], \
            f"eruption landed cervical to M1 under {orient}: {pct}"

    print("OK  molars run M3 -> M2 -> M1 tipward, eruption tipward of M1,")
    print("    and the volume's orientation flag does not change z_percent")
    print("\nMIRROR PASS")


def test_molars_from_the_tooth_split():
    """M1-M3 from the molar step fill in unpicked molars; a pick wins; each row keeps where it came from."""
    import json
    d = tempfile.mkdtemp()
    summary = {"n_teeth": 3, "teeth": [
        {"name": "M1", "landmark_z_start": 160, "landmark_z_end": 130},
        {"name": "M2", "landmark_z_start": 129, "landmark_z_end": 110},
        {"name": "M3", "landmark_z_start": 108, "landmark_z_end": 95}]}
    path = os.path.join(d, "S_molar_enamel.json")
    with open(path, "w") as f:
        json.dump(summary, f)
    molars = M.marks_from_molars(path)
    assert [(m["landmark"], m["z_start"], m["z_end"], m["source"]) for m in molars] == [
        ("M1", 160., 130., "teeth"), ("M2", 129., 110., "teeth"), ("M3", 108., 95., "teeth")]

    picked = [{"landmark": "eruption", "z_start": 40., "z_end": None, "source": "manual"},
              {"landmark": "M2", "z_start": 127., "z_end": 112., "source": "manual"}]
    marks = M.merge_molar_marks(picked, molars)
    assert {m["landmark"]: (m["z_start"], m["source"]) for m in marks} == {
        "eruption": (40., "manual"), "M2": (127., "manual"), "M1": (160., "teeth"), "M3": (108., "teeth")}

    rows = map_landmarks(tr, marks, "tip-high-z", tip_at_index_0=False)
    assert {r["landmark"]: r["source"] for r in rows} == {
        "eruption": "manual", "M1": "teeth", "M2": "manual", "M3": "teeth"}
    out = os.path.join(d, "lm.csv")
    write_csv(rows, out)
    assert {m["landmark"]: m["source"] for m in read_marks_csv(out)} == {r["landmark"]: r["source"] for r in rows}

    # Two teeth may be a fused pair under a molar's name: nothing is placed from them.
    with open(path, "w") as f:
        json.dump({**summary, "n_teeth": 2}, f)
    assert M.marks_from_molars(path) == []
    print("OK  molars from the tooth split: picks win, sources kept, two teeth place nothing")


test_spans_and_events()
test_curved_centerline()
test_centerline_traced_from_tip_not_mirrored()
test_molars_from_the_tooth_split()
print("\nALL TESTS PASS")
