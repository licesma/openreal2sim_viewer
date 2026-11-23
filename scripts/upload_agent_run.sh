#!/usr/bin/env bash
# Upload one agent reconstruction to s3://openreal2sim-sam/agent/<run>/.
# Keys mirror the pipeline layout so the viewer hooks work with the agent prefix:
#   reconstruction/scene.glb, source/000000.jpg
# plus the agent-only extras under agent/: render.png, scene.json, NOTES.md, objects/*.glb.
# Usage: scripts/upload_agent_run.sh <run> [run_dir]
set -euo pipefail

RUN="${1:?usage: upload_agent_run.sh <run> [run_dir]}"
RUN_DIR="${2:-/home/esteban/agent_vs_pipeline/$RUN}"
BUCKET="s3://openreal2sim-sam"
DEST="$BUCKET/agent/$RUN"

for f in scene.glb image.jpg scene.json render.png NOTES.md; do
  if [ ! -f "$RUN_DIR/$f" ]; then
    echo "missing $RUN_DIR/$f" >&2
    exit 1
  fi
done

aws s3 cp "$RUN_DIR/scene.glb" "$DEST/reconstruction/scene.glb" --content-type model/gltf-binary
aws s3 cp "$RUN_DIR/image.jpg" "$DEST/source/000000.jpg" --content-type image/jpeg
aws s3 cp "$RUN_DIR/render.png" "$DEST/agent/render.png" --content-type image/png
aws s3 cp "$RUN_DIR/scene.json" "$DEST/agent/scene.json" --content-type application/json
aws s3 cp "$RUN_DIR/NOTES.md" "$DEST/agent/NOTES.md" --content-type text/markdown
aws s3 cp "$RUN_DIR/objects/" "$DEST/agent/objects/" --recursive --exclude "*" --include "*.glb" --content-type model/gltf-binary

echo "uploaded $RUN to $DEST"
aws s3 ls "$DEST/" --recursive
