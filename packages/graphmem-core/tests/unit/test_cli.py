import json
import subprocess
import sys
import tempfile
from pathlib import Path

from graphmem.cli import _daemon_is_running, _daemon_pid_file, _daemon_start, _daemon_status, _daemon_stop, _parse_layers


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "graphmem.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_daemon_status_not_running():
    tmp = Path(tempfile.mkdtemp())
    assert _daemon_status(tmp) == 1


def test_daemon_start_stop():
    tmp = Path(tempfile.mkdtemp())
    # Start a fake daemon process (sleep)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_file = _daemon_pid_file(tmp)
    pid_file.write_text(str(proc.pid))

    assert _daemon_is_running(pid_file) is True
    assert _daemon_status(tmp) == 0

    _daemon_stop(tmp)
    assert not pid_file.exists()
    assert _daemon_is_running(pid_file) is False

    proc.kill()
    proc.wait()


def test_parse_layers():
    assert _parse_layers("L1,L2") == ("L1", "L2")
    assert _parse_layers("L1") == ("L1",)
    assert _parse_layers("L0,L1,L2,L3") == ("L0", "L1", "L2", "L3")
    assert _parse_layers(None) == ("L1", "L2")


def test_recall_json_output():
    tmp = tempfile.mkdtemp()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphmem.cli",
            "--home",
            tmp,
            "write",
            "user",
            "hello world",
            "--session-id",
            "s1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphmem.cli",
            "--home",
            tmp,
            "recall",
            "hello",
            "--layer",
            "L0",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "items" in data
    assert "formatted" in data
    assert "latency_ms" in data


def test_graph_depth_json():
    tmp = tempfile.mkdtemp()
    # Write a turn
    subprocess.run(
        [
            sys.executable,
            "-m",
            "graphmem.cli",
            "--home",
            tmp,
            "write",
            "user",
            "test content",
            "--session-id",
            "s2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Get node id
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphmem.cli",
            "--home",
            tmp,
            "recall",
            "test",
            "--layer",
            "L0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    node_id = data["items"][0]["node_id"]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphmem.cli",
            "--home",
            tmp,
            "graph",
            node_id,
            "--depth",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert "root_id" in data
    assert "nodes" in data
    assert "edges" in data
