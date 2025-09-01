#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "Working dir: $SCRIPT_DIR"

cd $SCRIPT_DIR

if [ ! -d "$SCRIPT_DIR/Wan2GP" ] ; then
  echo "Cloning Wan2GP"
  git clone https://github.com/deepbeepmeep/Wan2GP
else
  echo "Updating Wan2GP"
  cd $SCRIPT_DIR/Wan2GP
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

cd $SCRIPT_DIR

docker compose -f docker-compose.wan2gp.yml build


