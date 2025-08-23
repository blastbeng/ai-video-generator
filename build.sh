#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "Working dir: $SCRIPT_DIR"

cd $SCRIPT_DIR

if [ ! -d "$SCRIPT_DIR/framepack-studio" ] ; then
  echo "Cloning framepack-studio"
  git clone https://github.com/FP-Studio/framepack-studio
else
  echo "Updating framepack-studio"
  cd $SCRIPT_DIR/framepack-studio
  git pull
fi


if [ ! -d "$SCRIPT_DIR/MMAudio" ] ; then
  echo "Cloning MMAudio"
  git clone https://github.com/hkchengrex/MMAudio
else
  echo "Updating MMAudio"
  cd $SCRIPT_DIR/MMAudio
  git pull
fi

cd $SCRIPT_DIR

docker compose build


