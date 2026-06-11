#!/bin/sh
# Kairon web-app entrypoint. Ensures the persistent dirs are writable, then
# execs whatever CMD was passed (typically uvicorn).

set -eu

cd /app

mkdir -p /app/runs /app/data /app/reports

# On first start, the data dir is empty; nothing to seed. The first /upload
# request will create the runs.db on demand.

exec "$@"
