"""
End-to-end tests of the studio server, over real HTTP, with real jobs.

A server is started on a free port against a temporary workspace, and the
tests drive it the way the browser does: upload a scan in pieces, import it,
queue steps and wait for them. The jobs run the repository's own scripts on
the synthetic phantom, so a passing run means the studio's command lines,
output registration and dependency handling all agree with the tools as they
are today.

Run with:  python -m pytest studio/tests/test_server.py
"""

import http.client
import json
import math
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

import nrrd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import studio.server as server_module  # noqa: E402
from studio.server import clean, serve  # noqa: E402
from studio.tests.phantom import mandible, write_isq, write_segmentation  # noqa: E402
from studio.workspace import artifacts_in  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("studio")
    httpd, studio = serve(root / "ws", port=0, remember=False)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield {"port": httpd.server_address[1], "studio": studio, "data": root / "data"}
    studio.jobs.shutdown()
    httpd.shutdown()
    httpd.server_close()


def call(srv, method, path, body=None, headers=None, raw=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv['port']}{path}", method=method,
                                 data=raw if raw is not None else (json.dumps(body).encode() if body is not None else None),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            return resp.status, json.loads(data) if resp.headers.get_content_type() == "application/json" else data
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def wait_for(srv, predicate, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        jobs = call(srv, "GET", "/api/jobs")[1]
        if predicate(jobs):
            return jobs
        time.sleep(0.5)
    raise AssertionError("jobs did not settle in time")


def idle(jobs):
    return all(j["status"] in ("done", "failed", "cancelled") for j in jobs)


def test_preview_files_include_the_numbered_teeth():
    """The molar view's M1-M3 overlays are served; nothing that climbs out of the preview folder is."""
    match = server_module.VIEW_FILE.match
    assert match("ov_m1.u8.gz") and match("ov_enamel.u8.gz") and match("base.u16.gz")
    assert not match("ov_../m1.u8.gz") and not match("ov_M1.u8.gz") and not match("ov_m1.u8")


def test_state_and_static(server):
    status, state = call(server, "GET", "/api/state")
    assert status == 200 and state["steps"][0]["id"] == "convert"
    assert {t["id"] for t in state["tools"]} >= {"rsq_stability", "tests"}
    assert call(server, "GET", "/")[0] == 200
    assert call(server, "GET", "/js/app.js")[0] == 200
    assert call(server, "GET", "/js/missing.js")[0] == 404


def test_foreign_hosts_and_origins_are_refused(server):
    conn = http.client.HTTPConnection("127.0.0.1", server["port"], timeout=10)
    conn.request("GET", "/api/state", headers={"Host": f"attacker.example:{server['port']}"})
    assert conn.getresponse().status == 403
    conn.close()
    status, _ = call(server, "POST", "/api/jobs/clear", {}, headers={"Origin": "http://attacker.example"})
    assert status == 403


def test_upload_then_import(server):
    data = server["data"]
    data.mkdir(exist_ok=True)
    isq = write_isq(data / "upload_me.isq", shape=(60, 40, 30))
    status, up = call(server, "POST", "/api/uploads", {})
    assert status == 201
    payload = isq.read_bytes()
    status, res = call(server, "PUT", f"/api/uploads/{up['id']}/scan%201.isq", raw=payload,
                       headers={"Content-Type": "application/octet-stream"})
    assert status == 200 and res["bytes"] == len(payload)
    # A path that tries to climb out of the upload lands inside it instead.
    status, _ = call(server, "PUT", f"/api/uploads/{up['id']}/..%2F..%2Fescape.txt", raw=b"x",
                     headers={"Content-Type": "application/octet-stream"})
    assert status == 200
    ws_root = server["studio"].ws.root
    assert not (ws_root / "escape.txt").exists() and not (ws_root.parent / "escape.txt").exists()
    (ws_root / "uploads" / up["id"] / "escape.txt").unlink()

    status, info = call(server, "GET", f"/api/uploads/{up['id']}")
    assert info["kind"] == "isq"
    status, created = call(server, "POST", "/api/samples",
                           {"meta": {"age": "P52", "line": "Odam-Cre", "genotype": "WT"}, "source": {"upload": up["id"]},
                            "folder": "Uploads/Odam"})
    assert status == 201
    name = created["sample"]["name"]
    assert name == "P52_Odam-Cre_WT_1" and created["sample"]["folder"] == "Uploads/Odam"
    detail = call(server, "GET", f"/api/samples/{name}")[1]
    raw = next(a for a in detail["artifacts"] if a["key"] == "raw")
    assert raw["exists"] and "/raw/" in raw["path"]
    assert not (ws_root / "uploads" / up["id"]).exists()


def test_one_connection_serves_requests_whose_body_is_not_read(server):
    # The browser keeps one connection open and sends "{}" with every POST.
    # A reply that leaves those bytes unread makes them the start of the next
    # request on the connection ("{}PUT ..."), which http.server answers 501.
    conn = http.client.HTTPConnection("127.0.0.1", server["port"], timeout=10)
    json_body = {"Content-Type": "application/json"}
    binary = {"Content-Type": "application/octet-stream"}
    try:
        conn.request("POST", "/api/uploads", body=b"{}", headers=json_body)
        resp = conn.getresponse()
        assert resp.status == 201
        uid = json.loads(resp.read())["id"]
        sock = conn.sock
        conn.request("PUT", f"/api/uploads/{uid}/f0004019.nrrd", body=b"x" * 1000, headers=binary)
        resp = conn.getresponse()
        assert resp.status == 200 and json.loads(resp.read())["bytes"] == 1000
        # Refused before its body is read: the body must still be consumed.
        conn.request("PUT", "/api/uploads/0123456789abcdef/other.nrrd", body=b"y" * 5000, headers=binary)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404
        conn.request("POST", "/api/jobs/clear", body=b"{}", headers=json_body)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 200
        conn.request("GET", f"/api/uploads/{uid}")
        resp = conn.getresponse()
        assert resp.status == 200 and json.loads(resp.read())["files"] == 1
        assert conn.sock is sock, "the server dropped the kept-alive connection"
    finally:
        conn.close()
    call(server, "DELETE", f"/api/uploads/{uid}")


def test_quantify_a_linked_segmentation(server):
    folder = write_segmentation(server["data"] / "SEG", "P82_Odam-Cre_WT_1", seed=2)
    status, scan = call(server, "POST", "/api/import/scan", {"paths": [str(folder.parent)]})
    cand = next(c for c in scan["samples"] if c["name"] == "P82_Odam-Cre_WT_1")
    assert cand["meta"]["age"] == "P82"
    status, res = call(server, "POST", "/api/import/commit", {"samples": [cand]})
    assert res["created"] == ["P82_Odam-Cre_WT_1"]
    status, run = call(server, "POST", "/api/run", {"samples": ["P82_Odam-Cre_WT_1"], "steps": ["molar_quant"]})
    assert [j["step"] for j in run["jobs"]] == ["molar_quant"]
    jobs = wait_for(server, idle)
    molar = next(j for j in jobs if j["step"] == "molar_quant")
    assert molar["status"] == "done", molar["error"]
    summary = next(s for s in call(server, "GET", "/api/samples")[1] if s["name"] == "P82_Odam-Cre_WT_1")
    assert 0.1 < summary["metrics"]["enamel_fraction"] < 0.2
    views = {v["view"]: v for v in call(server, "GET", "/api/samples/P82_Odam-Cre_WT_1/views")[1]}
    assert views["molar"]["current"] and "enamel" in views["molar"]["overlays"]
    results = call(server, "GET", "/api/results")[1]
    assert "P82_Odam-Cre_WT_1" in results["molar"]
    status, csv_text = call(server, "GET", "/api/export/molar")
    assert status == 200 and csv_text.decode().startswith("sample,group,genotype,age,sex,enamel_fraction")


def test_the_landmarks_step_finds_them_and_keeps_hand_marks(server):
    """Run by itself after the straightening, the step places M1-M3 from the split; a hand mark keeps its place."""
    name = "P82_Odam-Cre_WT_3"
    folder = write_segmentation(server["data"] / "SEGMARKS", name, seed=4)
    status, scan = call(server, "POST", "/api/import/scan", {"paths": [str(folder.parent)]})
    cand = next(c for c in scan["samples"] if c["name"] == name)
    assert call(server, "POST", "/api/import/commit", {"samples": [cand]})[1]["created"] == [name]
    picked = [{"landmark": "eruption", "z_start": 40}, {"landmark": "M2", "z_start": 133, "z_end": 150}]
    assert call(server, "POST", f"/api/samples/{name}/marks", {"marks": picked, "run": False})[1]["saved"] == 2

    call(server, "POST", "/api/run", {"samples": [name], "steps": ["cmpr", "landmarks"]})
    jobs = wait_for(server, idle)
    job = next(j for j in jobs if j["step"] == "landmarks" and j["sample"] == name)
    assert job["status"] == "done", job["error"]
    assert {"--bone-volume", "--molar-volume", "--marks"} <= set(job["argv"]) and "--map-only" not in job["argv"]
    marks = call(server, "GET", f"/api/samples/{name}/marks")[1]
    placed = {m["landmark"]: (m["z_start"], m["z_end"], m["source"]) for m in marks["marks"]}
    # The phantom's crowns run z 96-127, 131-151 and 155-167.
    assert placed == {"eruption": (40.0, None, "manual"), "M2": (133.0, 150.0, "manual"),
                      "M1": (96.0, 127.0, "auto"), "M3": (155.0, 167.0, "auto")}
    # Where the step found each molar, the hand-marked one too, for the viewer to go back to.
    assert [(a["landmark"], a["z_start"], a["z_end"]) for a in marks["auto"] if a["landmark"] != "eruption"] == [
        (f"M{k}", z0, z1) for k, (z0, z1) in enumerate(((96.0, 127.0), (131.0, 151.0), (155.0, 167.0)), start=1)]
    assert {m["landmark"] for m in marks["mapped"]} == {"eruption", "M1", "M2", "M3"}
    assert {m["source"] for m in marks["mapped"] if m["landmark"] in ("eruption", "M2")} == {"manual"}

    # Saving from the viewer maps the marks as they are, without finding them again.
    saved = call(server, "POST", f"/api/samples/{name}/marks", {"marks": marks["marks"], "run": True})[1]
    assert saved["run"]["jobs"], saved
    jobs = wait_for(server, idle)
    remap = next(j for j in jobs if j["id"] == saved["run"]["jobs"][0]["id"])
    assert remap["status"] == "done", remap["error"]
    assert "--map-only" in remap["argv"] and "--bone-volume" not in remap["argv"]
    after = call(server, "GET", f"/api/samples/{name}/marks")[1]
    assert {m["landmark"]: m["source"] for m in after["marks"]} == {m: s for m, (_, _, s) in placed.items()}


def test_molar_density_is_converted_with_the_scans_calibration(server):
    """The conversion's metadata carries the ISQ's HA calibration to the molar step and the results."""
    name = "P82_Odam-Cre_WT_2"
    folder = write_segmentation(server["data"] / "SEGCAL", name, seed=3)
    status, scan = call(server, "POST", "/api/import/scan", {"paths": [str(folder.parent)]})
    cand = next(c for c in scan["samples"] if c["name"] == name)
    assert call(server, "POST", "/api/import/commit", {"samples": [cand]})[1]["created"] == [name]
    # The phantom's enamel sits near 950; a divisor of 170 puts it near 1900 mg HA/cm³.
    meta = server["data"] / "SEGCAL" / f"{name}_meta.json"
    meta.write_text(json.dumps({"mu_scaling": 170, "_filter_params": {"apply_rescale": False},
                                "calibration": {"slope": 375.507, "intercept": -197.288, "mu_scaling": 170}}))
    server["studio"].ws.set_artifacts(name, {"convert_meta": meta})

    params = {"molar_quant": {"percentile_low": 10, "percentile_high": 90}}
    call(server, "POST", "/api/run", {"samples": [name], "steps": ["molar_quant"], "params": params})
    jobs = wait_for(server, idle)
    job = next(j for j in jobs if j["step"] == "molar_quant" and j["sample"] == name)
    assert job["status"] == "done", job["error"]
    assert "--calibration" in job["argv"] and "--percentiles" in job["argv"]

    summary = call(server, "GET", f"/api/samples/{name}/molar")[1]["summary"]
    assert summary["density_percentiles"] == [10.0, 90.0]
    assert summary["enamel_low_raw"] < summary["enamel_mean_raw"] < summary["enamel_high_raw"]
    for stat in ("mean", "low", "high"):
        assert summary[f"enamel_{stat}_mgHA"] == pytest.approx(summary[f"enamel_{stat}_raw"] / 170 * 375.507 - 197.288)
    assert 1500 < summary["enamel_mean_mgHA"] < 2300
    assert Path(summary["calibration"]["source"]).name == meta.name

    results = call(server, "GET", "/api/results")[1]
    assert results["molar"][name]["enamel_mean_mgHA"] == pytest.approx(summary["enamel_mean_mgHA"])
    header, *rows = call(server, "GET", "/api/export/molar")[1].decode().splitlines()
    columns = header.split(",")
    assert {"enamel_mean_mgHA", "enamel_low_mgHA", "enamel_high_mgHA", "density_percentiles"} <= set(columns)
    row = next(dict(zip(columns, r.split(","))) for r in rows if r.startswith(name + ","))
    assert float(row["enamel_mean_mgHA"]) == pytest.approx(summary["enamel_mean_mgHA"])
    assert row["density_percentiles"] == "10/90" and row["calibration"] == meta.name

    # The molar view carries it too, so the compare page can put the heatmap in mg HA/cm³.
    status, view = call(server, "GET", f"/api/samples/{name}/views/molar/meta")
    assert status == 200 and view["calibration"]["divisor"] == 170
    assert view["calibration"]["slope"] == pytest.approx(375.507) and view["calibration"]["intercept"] == pytest.approx(-197.288)


def test_a_scan_import_reads_the_scans_and_names_the_batch(server):
    import datetime as dt
    import struct
    from urllib.parse import quote
    folder = server["data"] / "scanroom" / "P60"
    folder.mkdir(parents=True)
    for filename, day in (("C0005001.ISQ", 3), ("C0005002.ISQ", 4)):
        path = write_isq(folder / filename, shape=(20, 16, 12))
        header = bytearray(path.read_bytes())
        ticks = int((dt.datetime(2025, 2, day) - dt.datetime(1858, 11, 17)).total_seconds() * 10_000_000)
        struct.pack_into("<q", header, 36, ticks)
        path.write_bytes(bytes(header))

    status, scan = call(server, "POST", "/api/import/scan", {"paths": [str(folder.parent)]})
    found = {c["name"]: c for c in scan["samples"]}
    one, two = found["C0005001"], found["C0005002"]
    assert one["meta"]["scan_date"] == "2025-02-03" and not one["sources"]["scan_date"]["guess"]
    assert one["meta"]["age"] == "P60" and one["sources"]["age"]["guess"], "the folder's age is a guess"
    assert one["recorded"]["voxel_um"] == 12.0 and not one["exists"]
    # A single import reads the same details.
    info = call(server, "GET", f"/api/fs/inspect?path={quote(str(folder / 'C0005002.ISQ'))}")[1]
    assert info["meta"]["scan_date"] == "2025-02-04" and info["sources"]["age"]["from"] == "From the folder “P60”"

    details = {"line": "Odam-Cre", "genotype": "KO"}
    items = [{"key": c["name"], "found": c["name"], "meta": {**c["meta"], **details}} for c in (one, two)]
    status, plan = call(server, "POST", "/api/names/plan", {"items": items})
    assert [p["name"] for p in plan["names"]] == ["P60_Odam-Cre_KO_1", "P60_Odam-Cre_KO_2"]
    status, res = call(server, "POST", "/api/import/commit", {"folder": "Scan room", "samples": [
        {"found": c["name"], "name": "", "meta": {**c["meta"], **details}, "source": c["source"], "artifacts": c["artifacts"]}
        for c in (one, two)]})
    assert res["created"] == ["P60_Odam-Cre_KO_1", "P60_Odam-Cre_KO_2"], res["failed"]
    record = call(server, "GET", "/api/samples/P60_Odam-Cre_KO_2")[1]["record"]
    assert record["folder"] == "Scan room"
    assert record["meta"]["replicate"] == 2 and record["meta"]["scan_date"] == "2025-02-04"
    assert "voxel_um" not in record["meta"], "the file's own voxel size stays the file's"

    # Scanned again, both are known by their scans, under the names they were given.
    again = {c["name"]: c for c in call(server, "POST", "/api/import/scan", {"paths": [str(folder.parent)]})[1]["samples"]}
    assert again["C0005001"]["exists"] and again["C0005001"]["existing"] == "P60_Odam-Cre_KO_1"
    wait_for(server, idle)  # the source previews the import queued


def test_a_single_import_is_named_by_the_batch_rule(server):
    folder = server["data"] / "single"
    folder.mkdir(parents=True)
    write_isq(folder / "C0006001.ISQ", shape=(20, 16, 12))
    # An age and a line are not a name: P30_Odam-Cre_1 would read back as genotype Odam Cre.
    status, res = call(server, "POST", "/api/names/compose", {"meta": {"age": "P30", "line": "Odam-Cre"}, "found": "C0006001"})
    assert (res["name"], res["from"]) == ("C0006001", "found")
    status, res = call(server, "POST", "/api/names/compose", {"meta": {"age": "P30", "genotype": "WT"}, "found": "C0006001"})
    assert (res["name"], res["from"], res["replicate"]) == ("P30_WT_1", "details", 1)
    assert call(server, "POST", "/api/names/compose", {"meta": {}, "found": ""})[1]["name"] == ""
    status, res = call(server, "POST", "/api/names/compose", {"meta": {"age": "P30", "genotype": "WT", "sex": "M"}, "found": "C0006001"})
    assert res["name"] == "P30_WT_M_1"
    status, created = call(server, "POST", "/api/samples", {
        "meta": {"age": "P30", "line": "Odam-Cre"}, "source": {"path": str(folder / "C0006001.ISQ"), "mode": "link"}})
    assert status == 201 and created["sample"]["name"] == "C0006001"
    wait_for(server, idle)  # the source preview the import queued


def test_only_an_nrrd_skips_conversion(server):
    data = server["data"]
    data.mkdir(exist_ok=True)
    scan, _ = mandible(shape=(40, 24, 20))
    nrrd.write(str(data / "f0004019.nrrd"), scan.astype("int16"))
    tifffile.imwrite(data / "f0004020.tif", scan.astype("int16"))
    made = {}
    for filename, age in (("f0004019.nrrd", "P30"), ("f0004020.tif", "P31")):
        status, created = call(server, "POST", "/api/samples", {
            "meta": {"age": age, "line": "Odam-Cre", "genotype": "WT"},
            "source": {"path": str(data / filename), "mode": "link"}})
        assert status == 201
        made[filename] = created["sample"]
    nrrd_sample, tiff_sample = made["f0004019.nrrd"], made["f0004020.tif"]
    assert nrrd_sample["source"]["format"] == "nrrd" and not nrrd_sample["source"]["needs_conversion"]
    assert tiff_sample["source"]["format"] == "tiff" and tiff_sample["source"]["needs_conversion"]

    def pipeline(name):
        return {p["id"]: p for p in call(server, "GET", f"/api/samples/{name}")[1]["summary"]["pipeline"]}

    # An NRRD goes straight to segmentation; converting it is refused.
    steps_n = pipeline(nrrd_sample["name"])
    assert steps_n["convert"]["status"] == "skipped" and steps_n["segment"]["status"] == "ready"
    assert call(server, "GET", f"/api/samples/{nrrd_sample['name']}")[1]["default_run"][:1] == ["segment"]
    status, run = call(server, "POST", "/api/run", {"samples": [nrrd_sample["name"]], "steps": ["convert"]})
    assert not run["jobs"] and "Not needed" in run["skipped"][0]["reason"]
    # Every other format is converted before it is segmented.
    steps_t = pipeline(tiff_sample["name"])
    assert steps_t["convert"]["status"] == "ready"
    assert steps_t["segment"]["status"] == "blocked" and steps_t["segment"]["reason"] == "Convert the scan first"
    assert call(server, "GET", f"/api/samples/{tiff_sample['name']}")[1]["default_run"][:2] == ["convert", "segment"]
    wait_for(server, idle)  # the source previews the imports queued


def test_large_files_stay_beside_a_linked_scan(server):
    raw = server["data"] / "drive" / "RAW"
    raw.mkdir(parents=True)
    isq = write_isq(raw / "C0004030.ISQ")
    status, created = call(server, "POST", "/api/samples", {
        "meta": {"age": "P60", "line": "Odam-Cre", "genotype": "WT"}, "source": {"path": str(isq), "mode": "link"}})
    assert status == 201
    name = created["sample"]["name"]
    beside = raw / "MicroCT-Studio" / name
    storage = call(server, "GET", f"/api/samples/{name}")[1]["storage"]
    assert storage["path"] == str(beside) and not storage["exists"]
    # Copies left in the workspace by runs from before large files moved beside the scan.
    ws = server["studio"].ws
    folder = ws.sample_dir(name)
    stale = {"converted": folder / "converted" / f"{name}.nrrd",
             "reoriented": folder / "segmentation" / name / f"{name}_reoriented_volume.nrrd"}
    for path in stale.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")
    ws.set_artifacts(name, stale)

    # The lab's recipe, with one filter turned off for this run.
    status, run = call(server, "POST", "/api/run", {"samples": [name], "steps": ["convert", "segment", "molar_quant"],
                                                   "params": {"convert": {"nlm_h": 0.0}}})
    assert [j["step"] for j in run["jobs"]] == ["convert", "segment", "molar_quant"]
    jobs = wait_for(server, idle)
    mine = {j["step"]: j for j in jobs if j["sample"] == name and j["step"] != "preview"}
    assert all(j["status"] == "done" for j in mine.values()), {k: j["error"] for k, j in mine.items()}
    detail = call(server, "GET", f"/api/samples/{name}")[1]
    done = detail["record"]["steps"]["convert"]
    assert done["params"]["autocrop"] and done["params"]["reorient_tight"] and done["params"]["save_raw"]
    # Only what the run changed is kept for the next one to start from.
    assert done["custom"] == {"nlm_h": 0.0} and detail["record"]["steps"]["segment"]["custom"] == {}
    assert (beside / f"{name}_prefiltered.nrrd").exists() and (beside / f"{name}_prefiltered.studio.json").exists()
    arts = {a["key"]: a for a in detail["artifacts"]}
    assert arts["converted"]["path"] == str(beside / f"{name}.nrrd") and arts["converted"]["where"] == "storage"
    assert arts["reoriented"]["path"] == str(beside / f"{name}_reoriented_volume.nrrd")
    assert arts["reoriented"]["where"] == "storage"
    assert arts["incisor_volume"]["where"] == "workspace" and arts["molar_quant"]["where"] == "workspace"
    assert not any(path.exists() for path in stale.values())
    assert not list(folder.rglob("*_reoriented_volume.*")) and not list(folder.rglob(f"{name}.nrrd"))
    # Removing the specimen leaves its large files where they are, and says so.
    status, res = call(server, "DELETE", f"/api/samples/{name}")
    assert status == 200 and res["left_beside_scan"] == str(beside)
    assert (beside / f"{name}.nrrd").exists() and isq.exists()


def test_machine_learning_is_turned_off(server):
    state = call(server, "GET", "/api/state")[1]
    assert next(s for s in state["steps"] if s["id"] == "ml_segment")["disabled"]
    assert next(t for t in state["tools"] if t["id"] == "ml_train")["disabled"]
    folder = write_segmentation(server["data"] / "ML", "P40_Odam-Cre_WT_1", seed=3)
    name = server["studio"].ws.create({"age": "P40", "line": "Odam-Cre", "genotype": "WT"},
                                      artifacts=artifacts_in(folder, folder.name))["name"]
    status, run = call(server, "POST", "/api/run", {"samples": [name], "steps": ["ml_segment"]})
    assert not run["jobs"] and "turned off" in run["skipped"][0]["reason"]
    pipeline = {p["id"]: p for p in call(server, "GET", f"/api/samples/{name}")[1]["summary"]["pipeline"]}
    assert pipeline["ml_segment"]["status"] == "disabled"
    assert call(server, "POST", "/api/tools/ml_train", {"params": {}})[0] == 409
    assert call(server, "POST", "/api/tools/training_set", {})[0] == 409
    assert not (server["studio"].ws.root / "ml_training").exists()


def test_a_failed_step_cancels_what_depends_on_it(server):
    ws = server["studio"].ws
    broken = server["data"] / "broken_incisor_volume.nrrd"
    broken.write_bytes(b"not an nrrd")
    ws.create({"age": "P52", "genotype": "KO"}, name="Broken_1", artifacts={"incisor_volume": str(broken)})
    status, run = call(server, "POST", "/api/run", {"samples": ["Broken_1"], "steps": ["cmpr", "incisor_quant"]})
    assert [j["step"] for j in run["jobs"]] == ["cmpr", "incisor_quant"]
    jobs = wait_for(server, idle)
    by_step = {j["step"]: j for j in jobs if j["sample"] == "Broken_1"}
    assert by_step["cmpr"]["status"] == "failed"
    assert by_step["incisor_quant"]["status"] == "cancelled"
    pipeline = {p["id"]: p for p in call(server, "GET", "/api/samples/Broken_1")[1]["summary"]["pipeline"]}
    assert pipeline["cmpr"]["status"] == "failed"


def test_a_flag_leaves_a_specimen_out_with_its_note_until_it_is_included_again(server):
    name = "P40_Flagtest_WT_1"
    server["studio"].ws.create({"age": "P40", "line": "Flagtest", "genotype": "WT", "replicate": 1}, name=name)
    summary = lambda: next(s for s in call(server, "GET", "/api/samples")[1] if s["name"] == name)  # noqa: E731
    assert summary()["excluded"] is None

    status, _ = call(server, "POST", "/api/flags",
                     {"names": [name], "flag": {"note": "  Incisor tip\n broken ", "from": ["incisor"]}})
    assert status == 200
    flag = summary()["excluded"]
    assert (flag["from"], flag["note"]) == (["incisor"], "Incisor tip broken") and flag["at"]
    # Editing the note keeps the day it was flagged; naming every analysis means all of them.
    call(server, "POST", "/api/flags", {"names": [name], "flag": {"note": "Rescan", "from": ["segmentation", "molar", "incisor"]}})
    assert summary()["excluded"] == {"from": "all", "note": "Rescan", "at": flag["at"]}
    listed = next(s for s in call(server, "GET", "/api/results")[1]["samples"] if s["name"] == name)
    assert listed["excluded"]["note"] == "Rescan"
    notes = call(server, "GET", "/api/export/excluded")[1].decode()
    assert name in notes and "Rescan" in notes
    # The results page's CSV links name the specimens it shows.
    assert name in call(server, "GET", f"/api/export/metadata?s={name}")[1].decode()
    assert name not in call(server, "GET", "/api/export/metadata?s=Nobody_1,Nobody_2")[1].decode()

    assert call(server, "POST", "/api/flags", {"names": [name], "flag": {"from": []}})[0] == 400
    assert call(server, "POST", "/api/flags", {"names": [name], "flag": {"from": ["teeth"]}})[0] == 400
    assert call(server, "POST", "/api/flags", {"names": [], "flag": None})[0] == 400
    # A name that does not exist stops the whole request before anything is written.
    assert call(server, "POST", "/api/flags", {"names": [name, "Nobody_1"], "flag": None})[0] == 404
    assert summary()["excluded"]["note"] == "Rescan"

    assert call(server, "POST", "/api/flags", {"names": [name], "flag": None})[0] == 200
    assert summary()["excluded"] is None
    assert call(server, "DELETE", f"/api/samples/{name}")[0] == 200


def test_results_projects_over_the_api(server):
    status, made = call(server, "POST", "/api/results/projects", {"name": "Odam study", "names": ["A_1"], "by": ["genotype"]})
    assert status == 201 and made["names"] == ["A_1"]
    assert call(server, "POST", "/api/results/projects", {"name": "odam study"})[0] == 409
    assert call(server, "POST", "/api/results/projects", {"name": ""})[0] == 400
    status, changed = call(server, "PATCH", f"/api/results/projects/{made['id']}", {"names": ["A_1", "B_2"]})
    assert status == 200 and changed["names"] == ["A_1", "B_2"] and changed["name"] == "Odam study"
    assert call(server, "GET", "/api/settings")[1]["results_projects"][0]["names"] == ["A_1", "B_2"]
    assert call(server, "PATCH", "/api/results/projects/abc123", {"name": "x"})[0] == 404
    assert call(server, "DELETE", f"/api/results/projects/{made['id']}")[0] == 200
    assert call(server, "GET", "/api/settings")[1]["results_projects"] == []


def test_library_folders_file_specimens_over_the_api(server):
    ws = server["studio"].ws
    names = ["P41_Foldertest_WT_1", "P41_Foldertest_WT_2"]
    for n in names:
        ws.create({"age": "P41", "line": "Foldertest", "genotype": "WT"}, name=n)
    summary = lambda n: next(s for s in call(server, "GET", "/api/samples")[1] if s["name"] == n)  # noqa: E731
    assert summary(names[0])["folder"] == ""

    assert call(server, "POST", "/api/folders", {"folder": "Folder test/Batch 1"})[0] == 201
    assert call(server, "POST", "/api/folders", {"folder": "folder test/batch 1"})[0] == 409
    assert call(server, "POST", "/api/folders", {"folder": " / "})[0] == 400
    status, body = call(server, "POST", "/api/folders/move", {"names": names, "folder": "Folder test/Batch 1"})
    assert status == 200 and body["folder"] == "Folder test/Batch 1"
    assert summary(names[1])["folder"] == "Folder test/Batch 1"
    assert call(server, "POST", "/api/folders/move", {"names": [names[0], "Nobody_1"], "folder": ""})[0] == 404
    assert call(server, "POST", "/api/folders/move", {"names": [], "folder": ""})[0] == 400

    status, body = call(server, "POST", "/api/folders/rename", {"from": "Folder test", "to": "Folder test 2"})
    assert status == 200 and sorted(body["moved"]) == names
    assert summary(names[0])["folder"] == "Folder test 2/Batch 1"
    assert call(server, "POST", "/api/folders/rename", {"from": "Nowhere", "to": "Else"})[0] == 404
    assert call(server, "POST", "/api/folders/rename", {"from": "Folder test 2", "to": "Folder test 2/Batch 1/In"})[0] == 400
    assert "Folder test 2/Batch 1" in call(server, "GET", "/api/state")[1]["workspace"]["folders"]

    status, body = call(server, "POST", "/api/folders/remove", {"folder": "Folder test 2"})
    assert status == 200 and summary(names[0])["folder"] == "Batch 1"
    assert "Folder test 2" not in call(server, "GET", "/api/folders")[1]["folders"]
    assert call(server, "POST", "/api/folders/remove", {"folder": "Folder test 2"})[0] == 404
    call(server, "POST", "/api/folders/remove", {"folder": "Batch 1"})
    for n in names:
        assert summary(n)["folder"] == ""
        assert call(server, "DELETE", f"/api/samples/{n}")[0] == 200


def test_the_results_pages_colours_and_titles_are_kept_with_the_workspace(server):
    style = {"groups": {"genotype\u0001KO": "#d03b3b"}, "landmarks": {"M1": "#00aa00"}, "marks": {"axes": "#333333"},
             "titles": {"molar-density": "Molar enamel density"}, "labels": {"Enamel / dentin": "Enamel:dentin"}}
    status, settings = call(server, "PATCH", "/api/settings", {"results_style": style})
    assert status == 200 and settings["results_style"] == style
    assert call(server, "GET", "/api/state")[1]["workspace"]["settings"]["results_style"] == style
    # Replaced whole, so a colour reset back to its default is gone.
    del style["marks"]["axes"]
    assert call(server, "PATCH", "/api/settings", {"results_style": style})[1]["results_style"]["marks"] == {}
    assert call(server, "PATCH", "/api/settings", {"results_style": "red"})[0] == 400
    call(server, "PATCH", "/api/settings", {"results_style": {}})


def test_json_is_nan_safe():
    assert clean({"a": math.nan, "b": [1.0, math.inf], "c": {"d": 2.5}}) == {"a": None, "b": [1.0, None], "c": {"d": 2.5}}


def test_the_studio_starts_without_a_workspace_and_opens_one(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "find_workspaces", lambda: [])
    httpd, studio = serve(None, port=0, remember=True, prefs=tmp_path / "prefs.json")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    srv = {"port": httpd.server_address[1]}
    try:
        assert call(srv, "GET", "/api/state")[1]["workspace"] is None
        status, err = call(srv, "GET", "/api/samples")
        assert status == 409 and err["code"] == "no_workspace"
        # A folder that is not a workspace is not quietly turned into one.
        plain = tmp_path / "LabDrive"
        plain.mkdir()
        (plain / "notes.txt").write_text("scans from May")
        status, err = call(srv, "POST", "/api/workspace", {"path": str(plain)})
        assert status == 404 and err["code"] == "not_workspace" and err["empty"] is False
        status, _ = call(srv, "POST", "/api/workspace", {"parent": str(plain), "name": "MicroCT-Studio"})
        assert status == 200
        state = call(srv, "GET", "/api/state")[1]
        assert state["workspace"]["name"] == "MicroCT-Studio" and state["workspace"]["available"]
        assert call(srv, "GET", "/api/samples") == (200, [])
        recent = call(srv, "GET", "/api/workspaces")[1]["recent"]
        assert recent[0]["current"] and recent[0]["samples"] == 0
        assert json.loads((tmp_path / "prefs.json").read_text())["recent"][0]["id"] == state["workspace"]["id"]

        # The drive is unplugged: the studio says so instead of failing oddly.
        moved = tmp_path / "Mounted-Elsewhere"
        shutil.move(str(plain), str(moved))
        status, err = call(srv, "GET", "/api/samples")
        assert status == 409 and err["code"] == "workspace_missing"
        assert call(srv, "GET", "/api/state")[1]["workspace"]["available"] is False
        # ...and opening it where it is now works.
        status, _ = call(srv, "POST", "/api/workspace", {"path": str(moved / "MicroCT-Studio")})
        assert status == 200 and call(srv, "GET", "/api/samples")[0] == 200
    finally:
        studio.jobs.shutdown()
        httpd.shutdown()
        httpd.server_close()


def test_a_workspace_can_be_removed_from_the_list_and_put_back(tmp_path, monkeypatch):
    prefs = tmp_path / "prefs.json"
    roots = [tmp_path / name for name in ("Old", "Now", "OnDrive")]
    monkeypatch.setattr(server_module, "find_workspaces", lambda: [r for r in roots if r.is_dir()])
    httpd, studio = serve(None, port=0, remember=True, prefs=prefs)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    srv = {"port": httpd.server_address[1]}
    names = lambda key: [w["name"] for w in call(srv, "GET", "/api/workspaces")[1][key]]  # noqa: E731
    try:
        for root in roots:
            assert call(srv, "POST", "/api/workspace", {"path": str(root), "create": True})[0] == 200
        call(srv, "POST", "/api/workspace", {"path": str(roots[1])})
        assert names("recent") == ["Now", "OnDrive", "Old"]
        old = next(w for w in call(srv, "GET", "/api/workspaces")[1]["recent"] if w["name"] == "Old")

        # Off the recent list, and not straight back under "found on connected drives".
        status, undo = call(srv, "POST", "/api/workspaces/remove", {"path": old["path"], "id": old["id"]})
        assert status == 200 and [e["path"] for e in undo["removed"]] == [old["path"]]
        assert names("recent") == ["Now", "OnDrive"] and "Old" not in names("found")
        assert (roots[0] / "studio.json").exists()  # the folder is left alone

        # The open one stays.
        now = call(srv, "GET", "/api/workspaces")[1]["recent"][0]
        status, err = call(srv, "POST", "/api/workspaces/remove", {"path": now["path"], "id": now["id"]})
        assert status == 409 and names("recent")[0] == "Now"

        # Undo puts it back where it was.
        assert call(srv, "POST", "/api/workspaces/restore", undo)[0] == 200
        assert names("recent") == ["Now", "OnDrive", "Old"]

        # One that is only found on a drive can be hidden too; opening it shows it again.
        call(srv, "POST", "/api/workspaces/remove", {"path": str(roots[2])})
        assert "OnDrive" not in names("recent") + names("found")
        assert json.loads(prefs.read_text())["last"] == str(roots[1])
        call(srv, "POST", "/api/workspace", {"path": str(roots[2])})
        assert names("recent")[0] == "OnDrive" and not json.loads(prefs.read_text())["hidden"]
        assert call(srv, "POST", "/api/workspaces/remove", {})[0] == 400

        # Removing the one --last would reopen stops that, and undo brings it back.
        call(srv, "POST", "/api/workspace", {"path": str(roots[0])})
        call(srv, "POST", "/api/workspace", {"path": str(roots[1])})
        studio.ws = None  # as when the studio starts with nothing open
        undo = call(srv, "POST", "/api/workspaces/remove", {"path": str(roots[1])})[1]
        assert undo["last"] == str(roots[1]) and "last" not in json.loads(prefs.read_text())
        call(srv, "POST", "/api/workspaces/restore", undo)
        assert json.loads(prefs.read_text())["last"] == str(roots[1]) and names("recent")[0] == "Now"
    finally:
        studio.jobs.shutdown()
        httpd.shutdown()
        httpd.server_close()
