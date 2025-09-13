#!/usr/bin/env bash
set -euo pipefail
curl -sSf http://localhost:${API_PORT:-8000}/health | grep '"ok"'
