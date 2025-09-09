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

#cd $SCRIPT_DIR
#
#if [ ! -d "$SCRIPT_DIR/Wan2GP" ] ; then
#  echo "Cloning Wan2GP"
#  git clone https://github.com/deepbeepmeep/Wan2GP
#else
#  echo "Updating Wan2GP"
#  cd $SCRIPT_DIR/Wan2GP
#  git pull
#fi

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

if [ ! -d "$SCRIPT_DIR/mmaudio-api" ] ; then
  echo "Cloning mmaudio-api"
  git clone https://github.com/VladoPortos/mmaudio-api
else
  echo "Updating mmaudio-api"
  cd $SCRIPT_DIR/mmaudio-api
  git pull
fi

cd $SCRIPT_DIR

wget -nc -q -O "$SCRIPT_DIR/framepack-studio/loras/hyvideo_FastVideo_LoRA-fp8.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hyvideo_FastVideo_LoRA-fp8.safetensors"
#wget -nc -q -O "$SCRIPT_DIR/framepack-studio/loras/hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn.safetensors"

docker compose build


