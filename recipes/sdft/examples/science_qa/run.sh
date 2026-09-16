#!/usr/bin/env bash
# The SDFT training stack for the reference's Science Q&A split, then the
# training stream through run.py. Setup (once): see README. State goes to $RUN_DIR.
#
# The stack is left running and a healthy one is reused; `docker compose down`
# stops it. A trained stack is bound to its scenario, so a fresh run needs a
# fresh RUN_DIR (or a restart with REEF_SCENARIO changed).
set -euo pipefail
cd "$(dirname "$0")"
REEF_ROOT="$(cd ../../../.. && pwd)"

REEF_IMAGE="${REEF_IMAGE:-reef}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
RUN_DIR="${RUN_DIR:-$PWD/work}"
export REEF_SERVICE_URL="${REEF_SERVICE_URL:-http://127.0.0.1:28901}"
export REEF_SCENARIO="${REEF_SCENARIO:-sdft-science-qa}"
export SDFT_WORK_DIR="$RUN_DIR"
# The reference checkout, for its dataset and the protocol the README cites.
REFERENCE_DIR="$RUN_DIR/self-distillation"
REFERENCE_REPOSITORY="https://github.com/idanshen/Self-Distillation"
REFERENCE_COMMIT="d77573212fa0"
export SCIENCE_DATA_DIR="$REFERENCE_DIR/data/science_data"

# Prerequisites
command -v uv >/dev/null || { echo "run.sh: uv not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
docker image inspect "$REEF_IMAGE" >/dev/null 2>&1 \
    || { echo "run.sh: image $REEF_IMAGE not found (build docker/Dockerfile.reef)" >&2; exit 1; }
[ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ] \
    || { echo "run.sh: $MODEL_DIR/Qwen2.5-7B-Instruct not found (hf download Qwen/Qwen2.5-7B-Instruct)" >&2; exit 1; }
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/token" ] || openssl rand -hex 16 > "$RUN_DIR/token"
export REEF_TOKEN="$(cat "$RUN_DIR/token")"

# 1. The reference implementation at its pin: the dataset lives in its tree.
if [ ! -d "$SCIENCE_DATA_DIR" ]; then
    echo "==> [1/3] $REFERENCE_REPOSITORY at $REFERENCE_COMMIT"
    git clone --quiet "$REFERENCE_REPOSITORY" "$REFERENCE_DIR"
    git -C "$REFERENCE_DIR" checkout --quiet "$REFERENCE_COMMIT"
fi

# 2. Compose owns the stack's startup and readiness.
echo "==> [2/3] the reef stack at $REEF_SERVICE_URL"
(
    export REEF_IMAGE MODEL_DIR RUN_DIR REEF_ROOT
    docker compose up -d --wait
) || { echo "run.sh: the stack never became healthy; docker compose logs" >&2; exit 1; }

# 3. The stream, in an ephemeral uv environment (reef-client and the dataset reader).
echo "==> [3/3] the training stream (scenario $REEF_SCENARIO)"
uv run --no-project --python 3.12 --with reef-client --with datasets run.py "$@"
