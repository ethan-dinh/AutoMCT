"""
Serving over Tailscale: reading this machine's tailnet address and names, and
the server answering to those names and nothing else.

Run with:  python -m pytest studio/tests/test_remote.py
"""

import http.client
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio.remote import TailscaleError, parse_status  # noqa: E402
from studio.server import serve  # noqa: E402

RUNNING = {"BackendState": "Running",
           "Self": {"DNSName": "ethans-macbook-pro.tail1234.ts.net.", "HostName": "Ethan’s MacBook Pro",
                    "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"]}}


def test_running_tailnet_gives_address_and_both_names():
    net = parse_status(RUNNING)
    assert net.ip == "100.101.102.103"
    assert net.names == ["ethans-macbook-pro", "ethans-macbook-pro.tail1234.ts.net"]
    assert net.urls(8765)[0] == "http://100.101.102.103:8765/"


def test_without_magicdns_the_address_is_enough():
    net = parse_status({**RUNNING, "Self": {**RUNNING["Self"], "DNSName": ""}})
    assert net.names == [] and net.urls(9000) == ["http://100.101.102.103:9000/"]


@pytest.mark.parametrize("state, says", [("Stopped", "switch it on"), ("NeedsLogin", "sign in")])
def test_stopped_or_signed_out_says_what_to_do(state, says):
    with pytest.raises(TailscaleError, match=says):
        parse_status({"BackendState": state, "Self": {}})


def test_server_answers_to_tailnet_names_only(tmp_path):
    httpd, studio = serve(None, port=0, remember=False, prefs=tmp_path / "prefs.json",
                          extra_hosts=("Ethans-MacBook-Pro", "ethans-macbook-pro.tail1234.ts.net"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def status(host, method="GET", path="/api/state", origin=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Host": f"{host}:{port}", "Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        conn.request(method, path, body=b"{}" if method == "POST" else None, headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    try:
        code, body = status("ethans-macbook-pro")
        assert code == 200 and json.loads(body)["remote"] is False
        assert status("ethans-macbook-pro.tail1234.ts.net")[0] == 200
        assert status("localhost")[0] == 200
        assert status("other-machine.tail1234.ts.net")[0] == 403
        # A page served under the tailnet name may change things; one elsewhere may not.
        name = f"http://ethans-macbook-pro:{port}"
        assert status("ethans-macbook-pro", "POST", "/api/jobs/clear", origin="http://evil.example")[0] == 403
        assert status("ethans-macbook-pro", "POST", "/api/jobs/clear", origin=name)[0] != 403
    finally:
        if studio.jobs is not None:
            studio.jobs.shutdown()
        httpd.shutdown()
        httpd.server_close()
