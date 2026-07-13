#!/usr/bin/env python3
"""Deterministic end-to-end checks for Pebble's real terminal interface."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path


ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def plain(data: bytes) -> str:
    return ANSI.sub("", data.decode("utf-8", "replace")).replace("\r", "")


class Session:
    def __init__(self, command: list[str], env: dict[str, str], width: int = 56, height: int = 18):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        self.process = subprocess.Popen(
            command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            close_fds=True,
        )
        os.close(slave)
        self.output = b""

    def wait_for(self, text: str, timeout: float = 8.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rendered = plain(self.output)
            if text in rendered:
                return rendered
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if not ready:
                if self.process.poll() is not None:
                    break
                continue
            try:
                self.output += os.read(self.master, 8192)
            except OSError:
                break
        rendered = plain(self.output)
        raise AssertionError(f"timed out waiting for {text!r}\n--- terminal ---\n{rendered}")

    def wait_for_after(self, text: str, offset: int, timeout: float = 8.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rendered = plain(self.output[offset:])
            if text in rendered:
                return rendered
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if ready:
                try:
                    self.output += os.read(self.master, 8192)
                except OSError:
                    break
            elif self.process.poll() is not None:
                break
        raise AssertionError(
            f"timed out waiting for new {text!r}\n--- terminal ---\n{plain(self.output)}"
        )

    def send(self, data: bytes) -> None:
        os.write(self.master, data)

    def resize(self, width: int, height: int) -> None:
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        os.close(self.master)


def isolated_env(config_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PEBBLE_CONFIG_HOME"] = str(config_home)
    env["TERM"] = "xterm-256color"
    for key in (
        "NANOGPT_API_KEY",
        "NEURALWATT_API_KEY",
        "LILAC_API_KEY",
        "SYNTHETIC_API_KEY",
        "OPENCODE_GO_API_KEY",
        "XAI_API_KEY",
        "EXA_API_KEY",
    ):
        env.pop(key, None)
    return env


def seed_catalog(config_home: Path) -> None:
    model = {
        "service": "nanogpt",
        "info": {
            "id": "测试/模型-alpha",
            "object": "model",
            "created": 0,
            "owned_by": "pty-smoke",
            "name": "Unicode 模型",
        },
    }
    cache = {
        "services": {
            "nanogpt": {
                "updated_at": int(time.time()),
                "models": [model],
                "last_error": None,
            }
        }
    }
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "model-catalogs.json").write_text(json.dumps(cache), encoding="utf-8")


def check_repl_lifecycle(binary: Path, root: Path) -> None:
    config_home = root / "repl-config"
    seed_catalog(config_home)
    session = Session([str(binary)], isolated_env(config_home), width=48, height=16)
    try:
        screen = session.wait_for("Not connected")
        assert "/login nanogpt" in screen
        assert "<platform default>" not in screen

        session.send(b"discard this input")
        mark = len(session.output)
        session.send(b"\x03")
        session.wait_for_after("build ❯", mark)
        mark = len(session.output)
        session.send(b"/status\r")
        session.wait_for("Pebble status")
        session.wait_for_after("build ❯", mark)

        session.send(b"/login\r")
        session.wait_for("Connect a provider")
        mark = len(session.output)
        session.send(b"\r")
        session.wait_for_after("build ❯", mark)

        session.resize(36, 12)
        session.send(b"/model\r")
        session.wait_for("Choose a model")
        session.send("模型".encode("utf-8"))
        session.wait_for("Search: 模型")
        mark = len(session.output)
        session.send(b"\x1b")
        session.wait_for_after("Search: start typing", mark)
        mark = len(session.output)
        session.send(b"q")
        session.wait_for_after("build ❯", mark)

        session.send(b"\x04")
        session.wait_for("Use /resume to return")
        session.process.wait(timeout=3)
        assert session.process.returncode == 0
    finally:
        session.close()


def write_grok_stub(path: Path) -> None:
    path.write_text(
        r"""#!/bin/sh
while IFS= read -r line; do
case "$line" in
*'"method":"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"authMethods":[{"id":"cached_token"}]}}' ;;
*'"method":"authenticate"'*) echo '{"jsonrpc":"2.0","id":2,"result":{}}' ;;
*'"method":"session/new"'*) echo '{"jsonrpc":"2.0","id":3,"result":{"sessionId":"pty"}}' ;;
*'"method":"session/prompt"'*) echo '{"jsonrpc":"2.0","method":"session/update","params":{"update":{"sessionUpdate":"agent_message_chunk","content":{"text":"Streaming response arrived before completion. "}}}}'; sleep 30 ;;
esac
done
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def check_grok_interrupt(binary: Path, root: Path) -> None:
    config_home = root / "grok-config"
    stub = root / "grok-stub"
    write_grok_stub(stub)
    env = isolated_env(config_home)
    env["PEBBLE_GROK_CLI"] = str(stub)
    session = Session([str(binary), "--model", "grok/grok-test"], env)
    try:
        session.wait_for("grok-test")
        session.wait_for("build ❯")
        session.send(b"hello\r")
        session.wait_for("Streaming response arrived bef", timeout=5)
        mark = len(session.output)
        os.kill(session.process.pid, signal.SIGINT)
        session.wait_for("Cancelled.", timeout=5)
        session.wait_for_after("build ❯", mark, timeout=5)
        assert session.process.poll() is None
        session.send(b"discard after cancellation")
        mark = len(session.output)
        session.send(b"\x03")
        session.wait_for_after("build ❯", mark, timeout=5)
        session.send(b"/exit\r")
        session.wait_for("Use /resume to return")
        session.process.wait(timeout=3)
        assert session.process.returncode == 0
    finally:
        session.close()


def check_tool_interrupt(binary: Path, root: Path) -> None:
    config_home = root / "tool-config"
    stub = root / "grok-tool-stub"
    stub.write_text(
        r"""#!/bin/sh
while IFS= read -r line; do
case "$line" in
*'"method":"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"authMethods":[{"id":"cached_token"}]}}' ;;
*'"method":"authenticate"'*) echo '{"jsonrpc":"2.0","id":2,"result":{}}' ;;
*'"method":"session/new"'*) echo '{"jsonrpc":"2.0","id":3,"result":{"sessionId":"pty-tool"}}' ;;
*'"method":"session/prompt"'*) echo '{"jsonrpc":"2.0","method":"session/update","params":{"update":{"sessionUpdate":"agent_message_chunk","content":{"text":"<tool_call name=\"bash\"><arg name=\"command\">sleep 30</arg></tool_call>"}}}}'; echo '{"jsonrpc":"2.0","id":4,"result":{"stopReason":"tool_use"}}' ;;
esac
done
""",
        encoding="utf-8",
    )
    stub.chmod(0o700)
    env = isolated_env(config_home)
    env["PEBBLE_GROK_CLI"] = str(stub)
    session = Session(
        [
            str(binary),
            "--model",
            "grok/grok-test",
            "--permission-mode=danger-full-access",
        ],
        env,
    )
    try:
        session.wait_for("build ❯")
        session.send(b"run the slow command\r")
        session.wait_for("Working...", timeout=5)
        time.sleep(1.0)
        mark = len(session.output)
        os.kill(session.process.pid, signal.SIGINT)
        session.wait_for("Cancelled.", timeout=5)
        session.wait_for_after("build ❯", mark, timeout=5)
        assert session.process.poll() is None
        session.send(b"/exit\r")
        session.wait_for("Use /resume to return")
        session.process.wait(timeout=3)
        assert session.process.returncode == 0
    finally:
        session.close()


def check_plain_output(binary: Path, root: Path) -> None:
    config_home = root / "plain-config"
    for settings in (
        {"NO_COLOR": "1"},
        {"CLICOLOR": "0"},
        {"TERM": "dumb"},
    ):
        env = isolated_env(config_home)
        env.update(settings)
        completed = subprocess.run(
            [str(binary), "--help"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        assert b"\x1b" not in completed.stdout, (
            f"plain output contained ANSI escapes for {settings}: {completed.stdout!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/debug/pebble"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"Pebble binary not found: {binary}. Run `cargo build -p pebble` first.")

    root = Path(tempfile.mkdtemp(prefix="pebble-pty-smoke-"))
    try:
        check_repl_lifecycle(binary, root)
        check_grok_interrupt(binary, root)
        check_tool_interrupt(binary, root)
        check_plain_output(binary, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PTY smoke checks passed: onboarding, model/tool interrupts, resize, Unicode picker, plain output, login, and exit")


if __name__ == "__main__":
    main()
