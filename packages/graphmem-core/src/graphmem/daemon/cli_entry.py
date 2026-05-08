"""CLI entry point for graphmem daemon."""

import argparse

import uvicorn

from graphmem.daemon.main import create_app
from graphmem.memory import Memory


def main():
    parser = argparse.ArgumentParser(prog="graphmem-daemon")
    parser.add_argument("--home", required=True, help="graphmem home directory")
    parser.add_argument("--scope", required=True, help="memory scope")
    parser.add_argument("--port", type=int, default=8765, help="daemon port")
    parser.add_argument("--host", default="127.0.0.1", help="daemon host")
    args = parser.parse_args()

    mem = Memory.open(args.home, scope=args.scope)
    app = create_app(mem)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
