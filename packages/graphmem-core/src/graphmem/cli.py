"""CLI entry point for graphmem."""

import argparse
import sys
from pathlib import Path

from graphmem.memory import Memory


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

    graph_parser = subparsers.add_parser("graph", help="Show graph around a node")
    graph_parser.add_argument("node_id")

    subparsers.add_parser("compact", help="Run compression")
    subparsers.add_parser("stats", help="Show stats")
    subparsers.add_parser("doctor", help="Check health")

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

    mem = Memory.open(home, scope=scope)
    try:
        if args.command == "write":
            nid = mem.write_turn(args.role, args.content, session_id=args.session_id)
            print(nid)
        elif args.command == "recall":
            result = mem.recall(args.query, k=args.k)
            print(result.formatted)
        elif args.command == "graph":
            sg = mem.graph(args.node_id)
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
