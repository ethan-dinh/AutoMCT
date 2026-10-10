"""
Start MicroCT Studio.

    conda activate microct-analysis
    python -m studio                          # choose a workspace in the browser
    python -m studio --last                   # reopen the last one without asking
    python -m studio --workspace "/Volumes/LabDrive/MicroCT-Studio"
    python -m studio --port 8800 --no-browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

# numpy before torch anywhere in this process: macOS ships two OpenMP runtimes
# otherwise (see env/README.md). The server itself never imports torch, but
# jobs inherit this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .server import _load_prefs, serve  # noqa: E402
from .workspace import is_workspace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m studio", description="MicroCT Studio")
    parser.add_argument("--workspace", "-w", type=Path, default=None,
                        help="Open this workspace folder instead of choosing one in the browser")
    parser.add_argument("--last", action="store_true",
                        help="Reopen the last workspace used on this computer, if it is connected")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to listen on. Keep the default unless you know why not: "
                             "anyone who can reach the port can run jobs and browse files")
    parser.add_argument("--port", "-p", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()

    root = args.workspace
    if root is None and args.last:
        last = _load_prefs().get("last")
        root = Path(last) if last and is_workspace(Path(last)) else None
    server = studio = None
    for port in range(args.port, args.port + 20):
        try:
            server, studio = serve(root, host=args.host, port=port)
            break
        except OSError:
            continue
    if server is None:
        print(f"No free port between {args.port} and {args.port + 19}", file=sys.stderr)
        return 1

    host = "localhost" if args.host in ("127.0.0.1", "::1") else args.host
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"MicroCT Studio {url}")
    print(f"Workspace      {studio.ws.root if studio.ws else 'choose one in the browser'}")
    print("Stop with Ctrl-C. Running jobs are stopped with the studio.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        if studio.jobs is not None:
            studio.jobs.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
