#!/usr/bin/env bash
# Run an interactive, terminal-first demonstration of the spiral classifier.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ -x "$project_dir/.venv/bin/python" ]]; then
    python_bin="$project_dir/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    printf 'Python 3 is required. See %s/README.md\n' "$project_dir" >&2
    exit 1
fi

if ! "$python_bin" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
    printf 'NumPy and Matplotlib are missing. Install the requirements using the instructions in %s/README.md\n' \
        "$project_dir" >&2
    exit 1
fi

exec "$python_bin" -u "$project_dir/src/live_demo.py" "$@"
