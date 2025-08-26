#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR

docker compose down
$SCRIPT_DIR/build.sh

cd $SCRIPT_DIR
docker compose up -d