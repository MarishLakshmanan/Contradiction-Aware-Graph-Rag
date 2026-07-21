"""Starts the langgraph dev server and both Gradio UIs together for local development.

Run via `uv run dev` (registered in pyproject.toml's [project.scripts]).
"""

import subprocess
import sys

COMMANDS: list[list[str]] = [
    ["langgraph", "dev"],
    [sys.executable, "-m", "src.UI.agent_ui"],
    [sys.executable, "-m", "src.UI.pipeline_ui"],
]


def main() -> None:
    processes = [subprocess.Popen(cmd) for cmd in COMMANDS]
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
