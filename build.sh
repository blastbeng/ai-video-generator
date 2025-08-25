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

cd $SCRIPT_DIR

if [ ! -d "$SCRIPT_DIR/Fooocus" ] ; then
  echo "Cloning Fooocus"
  git clone https://github.com/lllyasviel/Fooocus.git
else
  echo "Updating Fooocus"
  cd $SCRIPT_DIR/Fooocus
  git pull
fi

cd $SCRIPT_DIR

if [ ! -d "$SCRIPT_DIR/Fooocus-API" ] ; then
  echo "Cloning Fooocus-API"
  git clone https://github.com/mrhan1993/Fooocus-API
else
  echo "Updating Fooocus-API"
  cd $SCRIPT_DIR/Fooocus-API
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


