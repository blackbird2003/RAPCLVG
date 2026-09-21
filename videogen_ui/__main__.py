from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m videogen_ui")
    subparsers = parser.add_subparsers(dest="command")
    start = subparsers.add_parser("start", help="Start the notebook web server")
    start.add_argument("--host", default=os.getenv("VIDEOGEN_HOST", "127.0.0.1"))
    start.add_argument("--port", type=int, default=int(os.getenv("VIDEOGEN_PORT", "7880")))
    start.add_argument(
        "--workspace",
        default=os.getenv("VIDEOGEN_WORKSPACE", ".runtime/videogen_ui"),
    )
    start.add_argument(
        "--runner",
        choices=("fake", "real"),
        default=os.getenv("VIDEOGEN_RUNNER", "real"),
        help="Execution backend. 'real' submits by default; set VIDEOGEN_REAL_SUBMIT=0 for dry-run.",
    )
    args = parser.parse_args()
    if args.command != "start":
        parser.print_help()
        raise SystemExit(2)
    os.environ["VIDEOGEN_WORKSPACE"] = args.workspace
    os.environ["VIDEOGEN_RUNNER"] = args.runner
    uvicorn.run("videogen_ui.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
