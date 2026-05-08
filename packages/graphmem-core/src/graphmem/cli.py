"""CLI entry point for graphmem."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from graphmem.memory import Memory
from graphmem.schema import Layer


def _daemon_pid_file(home: Path) -> Path:
    return home / "daemon.pid"


def _daemon_is_running(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, ProcessLookupError):
        return False


def _daemon_start(home: Path, scope: str, port: int) -> int:
    pid_file = _daemon_pid_file(home)
    if _daemon_is_running(pid_file):
        print("Daemon is already running.")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "graphmem.daemon.cli_entry",
        "--home",
        str(home),
        "--scope",
        scope,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid))
    print(f"Daemon started on port {port} (pid {proc.pid}).")
    return 0


def _daemon_stop(home: Path) -> int:
    pid_file = _daemon_pid_file(home)
    if not pid_file.exists():
        print("Daemon is not running.")
        return 0

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Daemon stopped (pid {pid}).")
    except (ValueError, OSError, ProcessLookupError):
        print("Daemon process not found, cleaning up pid file.")

    pid_file.unlink(missing_ok=True)
    return 0


def _daemon_status(home: Path) -> int:
    pid_file = _daemon_pid_file(home)
    if _daemon_is_running(pid_file):
        pid = int(pid_file.read_text().strip())
        print(f"Daemon is running (pid {pid}).")
        return 0
    else:
        print("Daemon is not running.")
        return 1


def _parse_layers(layer_str: str | None) -> tuple[Layer, ...]:
    if not layer_str:
        return (Layer.L1, Layer.L2)
    names = [name.strip().upper() for name in layer_str.split(",")]
    return tuple(Layer[name] for name in names)


def main():
    parser = argparse.ArgumentParser(prog="mem", description="graphmem CLI")
    parser.add_argument("--home", default="~/.graphmem", help="graphmem home directory")
    parser.add_argument("--scope", default="", help="memory scope")
    subparsers = parser.add_subparsers(dest="command")

    write_parser = subparsers.add_parser("write", help="Write a turn")
    write_parser.add_argument("role", choices=["user", "assistant"])
    write_parser.add_argument("content")
    write_parser.add_argument("--session-id", required=True)

    recall_parser = subparsers.add_parser("recall", help="Recall memories")
    recall_parser.add_argument("query")
    recall_parser.add_argument("--k", type=int, default=8)
    recall_parser.add_argument("--layer", default="L1,L2", help="Layers to search, e.g. L1,L2")
    recall_parser.add_argument("--json", action="store_true", help="Output JSON")

    graph_parser = subparsers.add_parser("graph", help="Show graph around a node")
    graph_parser.add_argument("node_id")
    graph_parser.add_argument("--depth", type=int, default=2, help="Traversal depth")
    graph_parser.add_argument("--json", action="store_true", help="Output JSON")

    subparsers.add_parser("compact", help="Run compression")
    subparsers.add_parser("stats", help="Show stats")
    subparsers.add_parser("doctor", help="Check health")

    daemon_parser = subparsers.add_parser("daemon", help="Daemon management")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_cmd")
    daemon_start = daemon_sub.add_parser("start", help="Start daemon")
    daemon_start.add_argument("--port", type=int, default=8765)
    daemon_sub.add_parser("stop", help="Stop daemon")
    daemon_sub.add_parser("status", help="Daemon status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    home = Path(args.home).expanduser()
    scope = args.scope or f"cli@{Path.home().name}:default"

    if args.command == "doctor":
        print(f"graphmem home: {home}")
        print(f"scope: {scope}")
        print(f"config exists: {(home / 'config.yaml').exists()}")
        return

    if args.command == "daemon":
        if args.daemon_cmd == "start":
            sys.exit(_daemon_start(home, scope, args.port))
        elif args.daemon_cmd == "stop":
            sys.exit(_daemon_stop(home))
        elif args.daemon_cmd == "status":
            sys.exit(_daemon_status(home))
        else:
            daemon_parser.print_help()
            sys.exit(1)

    mem = Memory.open(home, scope=scope)
    try:
        if args.command == "write":
            nid = mem.write_turn(args.role, args.content, session_id=args.session_id)
            print(nid)
        elif args.command == "recall":
            layers = _parse_layers(args.layer)
            result = mem.recall(args.query, k=args.k, layers=layers)
            if args.json:
                print(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "node_id": item.node.id,
                                    "layer": item.node.layer.value,
                                    "score": item.score,
                                }
                                for item in result.items
                            ],
                            "formatted": result.formatted,
                            "tokens": result.tokens,
                            "latency_ms": result.latency_ms,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                print(result.formatted)
        elif args.command == "graph":
            sg = mem.graph(args.node_id, depth=args.depth)
            if args.json:
                print(
                    json.dumps(
                        {
                            "root_id": sg.root_id,
                            "nodes": [
                                {
                                    "id": n.id,
                                    "layer": n.layer.value,
                                    "content": getattr(n, "content", "")[:200],
                                }
                                for n in sg.nodes
                            ],
                            "edges": [
                                {
                                    "id": e.id,
                                    "type": e.type.value,
                                    "from_id": e.from_id,
                                    "to_id": e.to_id,
                                }
                                for e in sg.edges
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                for n in sg.nodes:
                    print(f"[{n.layer.value}] {n.id}: {getattr(n, 'content', '')[:80]}")
        elif args.command == "compact":
            report = mem.compact()
            print(f"Episodes created: {report.episodes_created}")
        elif args.command == "stats":
            s = mem.stats()
            print(f"Scope: {s.scope}")
            for layer, count in s.nodes_by_layer.items():
                print(f"  {layer}: {count}")
    finally:
        mem.close()


if __name__ == "__main__":
    main()
