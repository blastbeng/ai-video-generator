import logging
import os
import random
import shutil
import json
import sys
import uuid
import asyncio
import requests
import urllib
import time
import re
import subprocess
import glob
import queue
import database
import multiprocessing
import concurrent
import cv2

from datetime import timedelta
from concurrent import futures
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask
from flask import send_file
from flask import Response
from flask import make_response
from flask import request
from flask import jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api
from flask_restx import Resource
from flask_restx import reqparse
from os.path import dirname
from os.path import join
from pathlib import Path
from threading import Timer
from flask_apscheduler import APScheduler
from gradio_client import Client, handle_file
from bs4 import BeautifulSoup, Tag

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)



logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')
        
log = logging.getLogger('werkzeug')
log.setLevel(int(os.environ.get("LOG_LEVEL")))

dbms = database.Database(database.SQLITE, dbname='configs.sqlite3')


class Config:    
    SECRET_KEY = os.environ.get("SECRET_KEY")

class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=10, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)

executor = ThreadPoolExecutorWithQueueSizeLimit(max_workers=1) 

LORAS = {}
LORAS["360c4m3r4_e100_only_double_blocks"] = ["360 face camera", "immersive portrait", "rotating view", "detailed expression"]
LORAS["CandidChange v2 e45"] = ["candid transition", "natural pose", "outdoor setting", "smooth scene change"]
LORAS["cinematik_flux50epoch"] = ["cinematic pan", "dynamic camera move", "dramatic landscape", "professional lighting"]
LORAS["cinematron"] = ["cinematic vibe", "versatile composition", "dramatic lighting", "storytelling shot"]
LORAS["hunyuan_darkraw"] = ["dramatic lighting", "high contrast", "shadowy figure", "intense spotlight"]
LORAS["kbk_backrooms_comfyui"] = ["backrooms style", "eerie hallways", "liminal space", "unsettling angles"]
LORAS["dolly-zoom-hunyuan-v1.0-vfx_ai"] = ["dolly zoom effect", "dramatic perspective shift", "focused subject", "blurred background"]
LORAS["HunyuanVideo - Glitchy DV Cam - Trigger is yellowjackets intro filmstyle"] = ["glitchy camcorder", "Yellowjackets style"]
LORAS["high-speed-drone-shot-hunyuan-v1.0-heavy-vfx_ai"] = ["high speed drone shot", "sweeping over a dense forest canopy at dusk", "fast aerial chase", "low altitude flyover of a rugged mountain range"]
LORAS["sidebyside_E75"] = ["synchronized angles", "multi-view scene", "dynamic action", "split-screen effect"]
LORAS["ultrawide_10_epochs"] = ["ultra wide angle", "cinematic perspective"]
LORAS["matrix-bullet-time-hunyuan-v1.0-vfx_ai"] = ["a rotating bullet-time shot"]
LORAS["hunyuan_mtv_grind2_600"] = ["90s MTV Grind", "energetic dance", "retro TV style", "vibrant crowd"]
LORAS["adapter_model"] = ["boxing match", "dynamic punch", "intense gym atmosphere", "muscular fighter"]
LORAS["boxing_epoch20"] = ["boxer in action", "intense fight scene", "dynamic punch pose", "gym background"]
LORAS["cywo1_Cyber_Woman_n1"] = ["cybernetic woman", "futuristic armor", "neon-lit cityscape", "high-tech visor"]
LORAS["defaultDance_preview"] = ["character performing default dance", "vibrant dance floor", "colorful lights", "energetic pose"]
LORAS["Digital_Human_Hunyuan"] = ["digital human", "pixelated skin", "futuristic interface", "glowing circuits"]
LORAS["dji_20250103_02-47-51_epoch9"] = ["Doge society", "humorous dog character", "industrial background", "playful pose"]
LORAS["DreamPunk_e33"] = ["dreampunk character", "dreamlike cyber city", "neon glow", "surreal pose"]
LORAS["closeupface-v1.1"] = ["female face portrait", "detailed skin texture", "realistic makeup", "soft lighting"]
LORAS["adapter_model_boxingE40"] = ["first person boxing", "POV punch", "intense ring action", "sweat and focus"]
LORAS["dabaichui_2"] = ["framepack dance", "natural dance flow", "vibrant stage", "rhythmic pose"]
LORAS["frostyfaces"] = ["frosty face", "icy features", "winter portrait", "cool blue tones"]
LORAS["idle_dance"] = ["dynamic movements", "energetic pose", "fluid action", "vibrant scene"]
LORAS["ph2t-h0n-v1.0"] = ["sexy dance Phut Hon style", "graceful movements", "nightclub setting"]
LORAS["poplock10"] = ["poplock dance", "rhythmic moves", "urban street", "robotic style"]
LORAS["pubg_146_framepack"] = ["PUBG 146", "blonde long hair", "playful dance", "expressive gestures", "lively arm waves"]
LORAS["Sexy_Dance_e15"] = ["sexy dance", "graceful pose", "dim lighting", "elegant flow"]
LORAS["adapter_modelsq"] = ["Squid Game guard", "pink suit", "menacing pose", "dystopian setting"]
LORAS["aifantasia"] = ["surreal human", "fantastical wings", "ethereal forest", "glowing aura"]
LORAS["executive_order_40_epochs"] = ["Trump signing order", "formal office", "authoritative pose", "official documents"]
LORAS["Tw3rk_e15"] = ["twerk dance", "dynamic hip movements", "energetic stage", "vibrant lighting"]
LORAS["UndergroundClub_hunyuan"] = ["underground club", "neon lights", "dancing crowd", "smoky atmosphere"]
LORAS["venom_hunyuan_video_v1_e40"] = ["Venom transformation", "dark symbiote", "intense expression", "urban rooftop"]
LORAS["kxsr_walking_anim_v1-5"] = ["character walking", "smooth gait", "urban street", "casual attire"]
LORAS["animal_documentary_epoch20"] = ["4n1m4l animal documentary", "sleek panther with neon green and electric purple outlines", "close-up of a scarlet macaw on a branch"]
LORAS["cat_epoch20"] = ["cat video", "short-haired tabby cat swiping at a butterfly", "orange kitten jumping at a toy feather"]
LORAS["Dog_epoch_40"] = ["dog video", "brown terrier walking down a sidewalk", "Shiba Inu with a superhero cape on a hill"]
LORAS["anaglyph 3D"] = ["anaglyph 3D", "red-cyan effect", "stereoscopic scene", "futuristic city"]
LORAS["Comic_Art_Illustration_Style.fp1600018"] = ["comic art", "bold lines", "vibrant colors", "illustrated scene"]
LORAS["fluidart-v1_hunyuanvideo_e28"] = ["fluid art", "abstract flow", "colorful swirls", "dynamic motion", "flu1dart", "fluidart style"]
LORAS["Graphical_Clothes_hyv"] = ["graphical clothes", "bold patterns", "artistic fabric", "fashion runway"]
LORAS["adapter_model_canvas"] = ["canvas art", "textured painting", "vibrant colors", "artistic scene"]
LORAS["adapter_model_watercolor"] = ["watercolor style", "soft brushstrokes", "fluid colors", "serene landscape"]
LORAS["Retro_Styles_Hunyuan"] = ["glitch art retro digital", "distorted pixel effect", "vibrant glitch overlay", "glitch art," "retro digital"]
LORAS["yuan-stars"] = ["Yuan style", "traditional-modern blend", "vibrant colors", "cultural scene"]
LORAS["hunyuan_20s_horror_900"] = ["1920s horror", "grainy film", "eerie mansion", "shadowy figure", "old black and white footage"]
LORAS["1950s_epoch50"] = ["1950s style", "vintage film look", "classic car chase", "retro cityscape"]
LORAS["hunyuan_kungfu_600"] = ["1970s martial arts", "classic kung fu pose", "retro dojo", "dynamic fight"]
LORAS["hunyuan_80s_fantasyv1_5_comfy"] = ["1980s fantasy movie style", "a brave knight wielding a glowing sword in a misty enchanted forest", "dramatic camera pan", "ethereal lighting," "1980s fantasy epic", "a sorceress casting a spell with colorful magic effects", "wide cinematic shot", "vintage film grain"]
LORAS["hunyuan_80s_horror_1000"] = ["1980s horror movie", "grainy film", "eerie mansion", "shadowy figure"]
LORAS["hunyuan_ancientrome"] = ["ancient Rome", "classical architecture", "Roman soldiers", "historic battle"]
LORAS["cyberpunk"] = ["cyberpunk neon cityscape", "futuristic night scene", "glowing holograms", "cyberpunk", "neon cityscape"]
LORAS["DigitalWave"] = ["digital wave", "futuristic patterns", "glowing circuits", "electronic flow"]
LORAS["xjx-TokyoRacerV2-comfy"] = ["highway racing Tokyo", "fast cars", "neon cityscape", "dynamic night drive", "highway racing", "Tokyo"]
LORAS["fxf-tokyoMeet-comfy"] = ["Tokyo meets", "urban fusion", "cultural blend", "vibrant streets"]
LORAS["parkour-freerunning-hunyuan-v1.0-vfx_ai"] = ["parkour jump", "fluid motion", "urban rooftop", "high-energy chase"]
LORAS["thanos-snap-r512-768-e20"] = ["character disintegrating", "dust effect", "dramatic fade", "post-battle scene"]
LORAS["GTA_epoch8"] = ["GTA 6 style", "vibrant city", "dynamic action", "car chase"]
LORAS["HeavyMetal512Epoch65"] = ["heavy metal style", "gritty texture", "bold leather", "stage performance"]
LORAS["RFX.XT404.V0.0.1"] = ["special effects explosion", "dramatic light burst", "cinematic action scene"]


def remove_directory_tree(start_directory: Path):
    """Recursively and permanently removes the specified directory, all of its
    subdirectories, and every file contained in any of those folders."""
    for path in start_directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            remove_directory_tree(path)

def text2img(params: dict) -> dict:
    host = os.environ.get("FOOOCUS_ENDPOINT")
    result = requests.post(url=f"{host}/v1/generation/text-to-image",
                           data=json.dumps(params),
                           headers={"Content-Type": "application/json"})
    return result.json()

def get_styles():
    host = os.environ.get("FOOOCUS_ENDPOINT")
    result = requests.get(url=f"{host}/v1/engines/styles",
                          headers={"Content-Type": "application/json"})
    return result.json()

def generate_image(config):
    result = text2img({
        "prompt": config["prompt_image"].replace("\n", " "),
        "negative_prompt": os.environ.get("NEGATIVE_PROMPT"),
        "performance_selection": "Quality",
        "aspect_ratios_selection": "704*1344",
        "guidance_scale": 20.0,
        "image_number": 1,
        "image_seed": config["image_seed"],
        "async_process": False,
        "style_selections": [ "Fooocus V2", "Fooocus V2 (Optional)", "Fooocus Enhance", "Fooocus Sharp", "Fooocus Negative", "Fooocus Cinematic", "Cinematic Diva", "Fooocus Photograph" ]
        })
    if len(result) > 0 and 'url' in result[0]:
        return result[0]['url']
    else:
        database.delete_wrong_entries(dbms)
        raise Exception("Result from Fooocus-API is None")

def download_file(url, extension, file_path=os.environ.get("OUTPUT_PATH")):
    full_path = file_path + str(uuid.uuid4()) + "." + extension
    urllib.request.urlretrieve(url, full_path)
    time.sleep(5)
    return full_path

def save_file(image, extension, file_path=os.environ.get("OUTPUT_PATH")):
    full_path = file_path + str(uuid.uuid4()) + "." + extension
    with open(full_path, "wb") as full_path_f:
        full_path_f.write(image)
    return full_path

def add_audio_to_video(file_path, config, prompt):
    video = cv2.VideoCapture(file_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    result_audio = Client(os.environ.get("FRAMEPACK_ENDPOINT")).predict(
        video_path={"video":handle_file(file_path)},
        prompt_text="",
        negative_text=os.environ.get("NEGATIVE_PROMPT"),
        variant_name="large_44k_v2",
        duration_sec=float(seconds),
        cfg_strength_val=7.0,
        steps_val=50,
        seed_val=random.randint(0, sys.maxsize),
        mask_away_clip_val=False,
        skip_video_composite_val=False,
        full_precision_val=False,
        api_name="/run_mmaudio"
    )
    if len(result_audio) >= 1 and 'value' in result_audio[1] and 'video' in result_audio[1]['value'] and config["status"] != 4:
        file_with_audio = os.environ.get("OUTPUT_PATH") + "audio/" + os.path.basename(result_audio[1]['value']['video'])
        return file_with_audio
    return None

def add_audio_to_video_old(file_path, config, prompt):
    url = os.environ.get("MMAUDIO_ENDPOINT") + "/process"
    video = cv2.VideoCapture(file_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    payload = {
        #'prompt': "the audio must necessarily use the Italian language in case there are people speaking, this is the prompt: '" + prompt + "'", 
        'negative_prompt': os.environ.get("NEGATIVE_PROMPT"), 
        'variant': "large_44k_v2", 
        'cfg_strenght': 7.0,
        'num_steps': 50, 
        #'duration': float(config["requested_seconds"]), 
        'duration': float(seconds), 
        #'seed': random.randint(0, 99999), 
        'mask_away_clip': False,
        'full_precision': True
    }
    with  open(file_path,'rb') as file:
        response = requests.post(url, data=payload, files={'video': file})
        response.raise_for_status()
        if response.status_code == 200:
            mmaudio_file_path = os.environ.get("OUTPUT_PATH") + os.path.basename(file_path).replace(".mp4","_MMAudio.mp4")
            with open(mmaudio_file_path, "wb") as mmaudio_f:
                mmaudio_f.write(response.content)
            return mmaudio_file_path
    return None

def remove_html_tags_and_content(text):
    soup = BeautifulSoup(text, features="html.parser")
    for tag in soup.find_all('think'):
        tag.replaceWith('')
    return soup.get_text() 

#def reset_workspace():
#    data = {
#        "message": "/reset",
#        "mode": "chat"
#    }
#    headers = {
#        'Authorization': 'Bearer ' + os.environ.get("ANYTHING_LLM_API_KEY")
#    }
#    anything_llm_url = os.environ.get("ANYTHING_LLM_ENDPOINT") + "/api/v1/workspace/" + os.environ.get("ANYTHING_LLM_WORKSPACE") + "/chat"
#    anything_llm_response = requests.post(url=anything_llm_url,
#                    data=data,
#                    headers=headers)
#    if (anything_llm_response.status_code != 200):
#        raise Exception("Error resetting AnythingLLM workspace")

def add_new_generation(video_len, mode, message, prompt, image, video, use_top):
    time.sleep(2)
    if database.select_config_by_skipped(dbms, 0) is None:

        if use_top: 
            config = database.select_top_config(dbms, mode, image, video)
            

        else:
            for n in range(10000):
                config = get_config(mode, image, video)
                value = database.select_config(dbms, config)
                if value is not None:
                    config["generation_id"] = value[0]
                    config["skipped"] =  value[1]
                    config["status"] =  value[2]
                if config["skipped"] is not None and config["skipped"] == 2:
                    logging.warn("Found skipped params: %s", str(config))
                else:
                    break

        if config["skipped"] is None or config["skipped"] == 0 or config["skipped"] == 1:
            config["exec_time_seconds"] = None
            config["requested_seconds"] = video_len
            
            if config["skipped"] is None:
                logging.warn("Saving params to database")
                config["generation_id"] = database.insert_wrong_config(dbms, config)
            config["skipped"] = 0
            config["status"] = 0
            database.update_config(dbms, config)
        
            config["prompt_image"] = None
            if prompt is None:
                #reset_workspace()
                message = (str(random.choice(json.loads(os.environ.get('PROMPT_LIST'))) if message is None else message))
                if"lora" in config and config["lora"] is not None:
                    for lora in config["lora"]:
                        lora_prompt = random.choice(LORAS[lora])
                        message = lora_prompt + ". " + message
                data = {
                    "message": message,
                    "mode": "chat"
                }
                headers = {
                    'Authorization': 'Bearer ' + os.environ.get("ANYTHING_LLM_API_KEY")
                }
                anything_llm_url = os.environ.get("ANYTHING_LLM_ENDPOINT") + "/api/v1/workspace/" + os.environ.get("ANYTHING_LLM_WORKSPACE") + "/chat"
                anything_llm_response = requests.post(url=anything_llm_url,
                                data=data,
                                headers=headers)
                
                if (anything_llm_response.status_code == 200):
                    
                    #prompt = remove_html_tags_and_content(anything_llm_response.json()["textResponse"].rstrip())
                    prompt = anything_llm_response.json()["textResponse"].rstrip()
                    if config["gen_photo"] == 1:
                        data_prompt_img = {
                            "message": 'Extract one scene this story, be synthetic, answer with just one sentence: "' + (str(prompt)) + '"',
                            "mode": "chat"
                        }
                        anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                                        data=data_prompt_img,
                                        headers=headers)
                        if (anything_llm_response_prompt_img.status_code == 200):
                            #config["prompt_image"] = remove_html_tags_and_content(anything_llm_response_prompt_img.json()["textResponse"].rstrip())
                            config["prompt_image"] = anything_llm_response_prompt_img.json()["textResponse"].rstrip()
                        else:
                            database.delete_wrong_entries(dbms)
                            raise Exception("Error getting response from AnythingLLM")
                    else:
                        config["prompt_image"] = prompt
                else:
                    database.delete_wrong_entries(dbms)
                    raise Exception("Error getting response from AnythingLLM")
            else:
                if config["gen_photo"] == 1:
                    data_prompt_img = {
                        "message": 'Extract the first sentence from this text: "' + (str(prompt)) + '"',
                        "mode": "chat"
                    }
                    headers = {
                        'Authorization': 'Bearer ' + os.environ.get("ANYTHING_LLM_API_KEY")
                    }
                    anything_llm_url = os.environ.get("ANYTHING_LLM_ENDPOINT") + "/api/v1/workspace/" + os.environ.get("ANYTHING_LLM_WORKSPACE") + "/chat"
                    anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                                    data=data_prompt_img,
                                    headers=headers)
                    if (anything_llm_response_prompt_img.status_code == 200):
                        #config["prompt_image"] = remove_html_tags_and_content(anything_llm_response_prompt_img.json()["textResponse"].rstrip())
                        config["prompt_image"] = anything_llm_response_prompt_img.json()["textResponse"].rstrip()
                    else:
                        database.delete_wrong_entries(dbms)
                        raise Exception("Error getting response from AnythingLLM")
                else:
                    config["prompt_image"] = prompt

            if "lora" in config and config["lora"] is not None:
                for lora in config["lora"]:
                    lora_prompt = random.choice(LORAS[lora])
                    prompt = prompt + ("" if prompt.endswith(".") else ".") + lora_prompt
                    config["prompt_image"] = config["prompt_image"] + ("" if config["prompt_image"].endswith(".") else ".") + lora_prompt

            photo_init = None
            video_init = None
            if image is not None:
                photo_init = save_file(image, ".png")
            elif video is not None:
                video_init = save_file(video, ".mp4")
            elif config["gen_photo"] == 1:
                start_image = generate_image(config)
                photo_init = download_file(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"), "png")

            photo_end = None
            if config["project"] == "framepack":
                mp4, config = get_video_framepack(mode, photo_init, video_init, config, prompt)
            elif config["project"] == "wan2gp":
                mp4, config = get_video_wan2gp(mode, photo_init, video_init, config, prompt)
            return mp4, config
        else:
            logging.error("I haven't found any working config")
    else:
        return False, None
    return None, None

def round_nearest(x, a, precision):
    return round((round(x / a) * a), precision)

def get_config(mode, image, video):
    config = {}
    gen_photo = (random.randint(0, 1)) if image is None else 0
    #gen_photo = 1 if image is None else 0
    model = ""
    if mode == 0 or mode == 1:
        model = "F1" if mode == 1 else "Original"
        if video is not None:
            model = "Video F1" if mode == 1 else "Video"
    elif mode == 2:
        model = "image2video" if gen_photo == 1 else "text2video"
        if video is not None:
            model = "video2video"
    config["project"] = "wan2gp" if mode == 2 else "framepack"
    config["model"] = model
    config["has_input_image"] = 1 if image is not None or gen_photo == 1 else 0
    config["has_input_video"] = 1 if video is not None else 0
    config["requested_seconds"] = None
    config["seed"] = random.randint(0, 999999999) if mode == 2 else random.randint(0, 21474)
    config["window_size"] = 129 if mode == 2 else 9
    config["steps"] = random.randint(3, 10) if mode == 2 else int(random.randrange(25, 41, 5))
    #config["steps"] = random.randint(20, 40)
    #config["steps"] = 25
    #config["cache_type"] = random.choice(["", "mag"]) if mode == 2 else random.choice(["None", "MagCache"])
    #config["cache_type"] = random.choice(["", "mag"]) if mode == 2 else random.choice(["MagCache","TeaCache"]) 
    config["cache_type"] = "MagCache"
    #config["cache_type"] = "None"
    config["tea_cache_steps"] = None if mode == 2 else int(random.randrange(5, 51, 5)) if config["cache_type"] == "TeaCache" else None
    config["tea_cache_rel_l1_thresh"] = None if mode == 2 else round_nearest(round(random.uniform(0.01, 1), 2), 0.05, 2) if config["cache_type"] == "TeaCache" else None
    config["mag_cache_threshold"] = None if mode == 2 else round_nearest(round(random.uniform(0.01, 1), 2), 0.05, 2) if config["cache_type"] == "MagCache" else None
    config["mag_cache_max_consecutive_skips"] = None if mode == 2 else random.randint(1, 5) if config["cache_type"] == "MagCache" else None
    config["mag_cache_retention_ratio"] = None if mode == 2 else round_nearest(round(random.uniform(0, 1), 2), 0.05, 2) if config["cache_type"] == "MagCache" else None
    config["distilled_cfg_scale"] = None if mode == 2 else round_nearest(round(random.uniform(1.0, 32), 1), 0.5, 1)
    config["cfg_scale"] = round_nearest(round(random.uniform(1.0, 20), 1), 0.5, 1) if mode == 2 else round(random.uniform(1, 3), 1) #1
    config["cfg_rescale"] = None if mode == 2 else round_nearest(round(random.uniform(0, 1), 2), 0.05, 2) #0
    
    lora_weights = ['0'] * len(LORAS)

    lora_list = []

    for idx in range(len(LORAS)):
        lora_weights[idx] = str(round_nearest(round(random.uniform(0.05, 2), 2), 0.05, 2))

    #framepack_resolution = random.choice([[384,640],[512,768]])
    framepack_resolution = [384,640]

    output_loras = random.sample(list(LORAS.keys()), random.randint(2, 10))
    #output_loras = random.choice(list(LORAS.keys()))

    
    config["lora"] = None #config["lora"] = output_loras
    config["lora_weight"] = lora_weights
    config["gen_photo"] = gen_photo
    config["skipped"] = None
    config['exec_time_seconds'] = None
    config['status'] = 0
    config['width'] = 720 if mode == 2 else framepack_resolution[0]
    config['height'] = 1280 if mode == 2 else framepack_resolution[1]
    config['top_config'] = 0
    config['upscale_model'] = "RealESRNet_x4plus"
    #config['upscale_model'] = random.choice(["RealESR-general-x4v3", "RealESRNet_x4plus", "RealESRGAN_x4plus"])
    config["image_seed"] = 0 if gen_photo == 0 else random.randint(0, sys.maxsize)
    return config

def start_video_gen_framepack(client, config, photo_init, video_init, prompt):
    result = client.predict(
        selected_model=config["model"],
        param_1=handle_file(photo_init) if photo_init is not None else None,
        param_2=({"video":handle_file(video_init)}) if video_init is not None else None,
        param_3=None,
        param_4=1,
        param_5=prompt.replace("\n", " "),
        param_6=os.environ.get("NEGATIVE_PROMPT"),
        param_7=config["seed"],
        param_8=False,
        param_9=config["requested_seconds"],
        param_10=config["window_size"], # window size
        param_11=config["steps"],
        param_12=config["cfg_scale"],
        param_13=config["distilled_cfg_scale"],
        param_14=config["cfg_rescale"],
        param_15=config["cache_type"],
        param_16=config["tea_cache_steps"],
        param_17=config["tea_cache_rel_l1_thresh"],
        param_18=config["mag_cache_threshold"],
        param_19=config["mag_cache_max_consecutive_skips"],
        param_20=config["mag_cache_retention_ratio"],
        param_21=4,
        param_22="Noise",
        param_23=True,
        param_24=config["lora"] if "lora" in config and config["lora"] is not None and len(config["lora"]) > 0 else [],
        param_25=config['width'],
        param_26=config['height'],
        param_27=True,
        param_28=5,
        param_30=float(config["lora_weight"][0]),
		param_31=float(config["lora_weight"][1]),
		param_32=float(config["lora_weight"][2]),
		param_33=float(config["lora_weight"][3]),
		param_34=float(config["lora_weight"][4]),
		param_35=float(config["lora_weight"][5]),
		param_36=float(config["lora_weight"][6]),
		param_37=float(config["lora_weight"][7]),
		param_38=float(config["lora_weight"][8]),
		param_39=float(config["lora_weight"][9]),
		param_40=float(config["lora_weight"][10]),
		param_41=float(config["lora_weight"][11]),
		param_42=float(config["lora_weight"][12]),
		param_43=float(config["lora_weight"][13]),
		param_44=float(config["lora_weight"][14]),
		param_45=float(config["lora_weight"][15]),
		param_46=float(config["lora_weight"][16]),
		param_47=float(config["lora_weight"][17]),
		param_48=float(config["lora_weight"][18]),
		param_49=float(config["lora_weight"][19]),
		param_50=float(config["lora_weight"][20]),
		param_51=float(config["lora_weight"][21]),
		param_52=float(config["lora_weight"][22]),
		param_53=float(config["lora_weight"][23]),
		param_54=float(config["lora_weight"][24]),
		param_55=float(config["lora_weight"][25]),
		param_56=float(config["lora_weight"][26]),
		param_57=float(config["lora_weight"][27]),
		param_58=float(config["lora_weight"][28]),
		param_59=float(config["lora_weight"][29]),
		param_60=float(config["lora_weight"][30]),
		param_61=float(config["lora_weight"][31]),
		param_62=float(config["lora_weight"][32]),
		param_63=float(config["lora_weight"][33]),
		param_64=float(config["lora_weight"][34]),
		param_65=float(config["lora_weight"][35]),
		param_66=float(config["lora_weight"][36]),
		param_67=float(config["lora_weight"][37]),
		param_68=float(config["lora_weight"][38]),
		param_69=float(config["lora_weight"][39]),
		param_70=float(config["lora_weight"][40]),
		param_71=float(config["lora_weight"][41]),
		param_72=float(config["lora_weight"][42]),
		param_73=float(config["lora_weight"][43]),
		param_74=float(config["lora_weight"][44]),
		param_75=float(config["lora_weight"][45]),
		param_76=float(config["lora_weight"][46]),
		param_77=float(config["lora_weight"][47]),
		param_78=float(config["lora_weight"][48]),
		param_79=float(config["lora_weight"][49]),
		param_80=float(config["lora_weight"][50]),
		param_81=float(config["lora_weight"][51]),
		param_82=float(config["lora_weight"][52]),
		param_83=float(config["lora_weight"][53]),
		param_84=float(config["lora_weight"][54]),
		param_85=float(config["lora_weight"][55]),
		param_86=float(config["lora_weight"][56]),
		param_87=float(config["lora_weight"][57]),
		param_88=float(config["lora_weight"][58]),
		param_89=float(config["lora_weight"][59]),
		param_90=float(config["lora_weight"][60]),
		param_91=float(config["lora_weight"][61]),
		#param_92=float(config["lora_weight"][62]),
		#param_93=float(config["lora_weight"][63]),
		#param_94=float(config["lora_weight"][64]),
		#param_95=float(config["lora_weight"][65]),
        api_name="/handle_start_button"
    )
    return result

def get_latest_file(path):
    list_of_files = glob.glob(path+'/*/*.png', recursive=True) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def start_video_gen_wan2gp(client, config, photo_init, video_init, prompt):
    client = Client("http://localhost:7860")
    result = client.predict(
        value=[{"image":handle_file(photo_init),"caption":None}],
        api_name="/_on_upload"
    )

    result = client.predict(
        value=[{"image":handle_file(photo_init),"caption":None}],
        api_name="/_on_gallery_change"
    )

    result = client.predict(
        target="settings",
        image_mask_guide={"background":None,"layers":[],"composite":None},
        lset_name="",
        image_mode=0,
        prompt=prompt.replace('\n',' ').replace('\r\n',' ').replace('\n\r',' '),
        negative_prompt="",
        resolution=str(config['width']) + "x" + str(config['height']),
        video_length=config["requested_seconds"]*24,
        batch_size=1,
		seed=config["seed"],
		force_fps="",
		num_inference_steps=config["steps"],
		guidance_scale=config["cfg_scale"],
        guidance2_scale=5,
        guidance3_scale=5,
        switch_threshold=0,
        switch_threshold2=0,
        guidance_phases=1,
        model_switch_phase=1,
        audio_guidance_scale=4,
        flow_shift=3,
        sample_solver="unipc",
        embedded_guidance_scale=6,
        repeat_generation=1,
        multi_prompts_gen_type=0,
        multi_images_gen_type=0,
        skip_steps_cache_type="mag" if config["cache_type"] is not None and config["cache_type"] == "MagCache" else None,
        skip_steps_multiplier=1.5,
        skip_steps_start_step_perc=20,
        loras_choices=[],
        loras_multipliers="",
        image_prompt_type="S",
        #image_start=[{"image":handle_file(os.environ.get("WAN2GP_ENDPOINT") + '/gradio_api/file=/tmp/gradio/ea3cd471bc75066f639e143f92698ae85dfa8ab3434470f2c3bd909474700ce5/image.png'),"caption":None}],  
        image_start=[{"image":handle_file('http://localhost:7860/gradio_api/file='+get_latest_file('/tmp/gradio')),"caption":None}],  
        image_end=[],
        model_mode=None,
        video_source=None,
        keep_frames_video_source="",
        video_guide_outpainting="#",
        video_prompt_type="",
        image_refs=[],
        frames_positions="",
        video_guide=None,
        image_guide=None,
        keep_frames_video_guide="",
        denoising_strength=0.5,
        video_mask=None,
        image_mask=None,
        control_net_weight=1,
        control_net_weight2=1,
        control_net_weight_alt=1,
        mask_expand=0,
        audio_guide=None,
        audio_guide2=None,
        audio_source=None,
        audio_prompt_type="V",
        speakers_locations="0:45 55:100",
        sliding_window_size=config["window_size"],
        sliding_window_overlap=1,
        sliding_window_color_correction_strength=0,
        sliding_window_overlap_noise=0,
        sliding_window_discard_last_frames=0,
        image_refs_relative_size=50,
        remove_background_images_ref=0,
        temporal_upsampling="",
        spatial_upsampling="",
        film_grain_intensity=0,
        film_grain_saturation=0.5,
        MMAudio_setting=0,
        MMAudio_prompt="",
        MMAudio_neg_prompt="",
        RIFLEx_setting=0,
        NAG_scale=1,
        NAG_tau=3.5,
        NAG_alpha=0.5,
        slg_switch=0,
        slg_layers=[9],
        slg_start_perc=10,
        slg_end_perc=90,
        apg_switch=0,
        cfg_star_switch=0,
        cfg_zero_step=-1,
        prompt_enhancer="",
        min_frames_if_references=1,
        override_profile=-1,
        mode="",
        api_name="/save_inputs_11"
        )

    result = client.predict(
        model_choice=None,
        api_name="/process_prompt_and_add_tasks"
        )

    result = client.predict(
        api_name="/prepare_generate_video"
        )

    result = client.predict(
        api_name="/activate_status"
        )

    result = client.predict(
        api_name="/process_tasks"
        )

    result = client.predict(
        api_name="/refresh_status_async"
        )

    result = client.predict(
        api_name="/refresh_gallery"
        )

    result = client.predict(
        api_name="/refresh_preview"
        )

    result = client.predict(
        input_file_list=None,
        api_name="/select_video"
        )
    return result

def monitor_job_framepack(client, job_id):
    monitor_result = client.predict(
            job_id=job_id,
            api_name="/monitor_job"
    )
    return monitor_result


def get_video_framepack(mode, photo_init, video_init, config, prompt):
    start = time.time()
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    logging.warn("Launching with params: %s", str(config))
    gen_result = start_video_gen_framepack(client, config, photo_init, video_init, prompt)
    try:
        
        config = database.select_config_by_id(dbms, config["generation_id"])
        if config["status"] != 4:
            config["status"] = 1
            database.update_config(dbms, config)
            if gen_result is not None and len(gen_result) > 0 and gen_result[1] is not None and gen_result[1]:
                job_id = gen_result[1]
                monitor_future = executor.submit(monitor_job_framepack, client, job_id)
                monitor_result = None
                
                try:
                    c_timeout = (config["requested_seconds"]*600)
                    if "lora" in config and config["lora"] is not None and len(config["lora"]) != 0:
                        logging.warn("Lora detected, adding some timeout to allow Lora loading")
                        c_timeout = c_timeout + (len(config["lora"]) * 90)
                    logging.warn("Using timeout: %s", str(c_timeout))
                    monitor_result = monitor_future.result(timeout=c_timeout)
                    monitor_future.cancel()
                except (concurrent.futures.TimeoutError, concurrent.futures._base.CancelledError) as e:
                    logging.error("Max Execution Time reached")
                    logging.error("Stopping current generation")
                    result_stop = client.predict(api_name="/end_process_with_update")
                    while True:
                        result_current = client.predict(api_name="/check_for_current_job")
                        if result_current is None or len(result_current) == 0 or (len(result_current) > 0 and (result_current[0] is None or result_current[0] == "")):
                            break
                        else:
                            time.sleep(60)
                            result_stop = client.predict(api_name="/end_process_with_update")
                    logging.error("Updating skipped param to 2 to database for id: " + str(config["generation_id"]))
                    end = time.time()
                    config["exec_time_seconds"] = int(end - start)
                    config["skipped"] = 2
                    config["status"] = 4
                    database.update_config(dbms, config)
                    raise(e)
                
                if monitor_result is not None and len(monitor_result) > 0 and 'video' in monitor_result[0]:
                    generated_video = (os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video']))
                    return upscale_and_add_audio(generated_video, config, prompt, start)
    
    except Exception as e:
        logging.error("Error!")
        logging.error("Updating skipped param to 2 to database for id: " + str(config["generation_id"]))
        end = time.time()
        config["exec_time_seconds"] = int(end - start)
        config["skipped"] = 2
        config["status"] = 4
        database.update_config(dbms, config)
        raise(e)
    database.delete_wrong_entries(dbms)
    return None, None


def get_video_wan2gp(mode, photo_init, video_init, config, prompt):
    start = time.time()
    client = Client(os.environ.get("WAN2GP_ENDPOINT"))
    logging.warn("Launching with params: %s", str(config))
    gen_result = start_video_gen_wan2gp(client, config, photo_init, video_init, prompt.rstrip())

    config = database.select_config_by_id(dbms, config["generation_id"])
    if config["status"] != 4:
        config["status"] = 1
        database.update_config(dbms, config)
        if gen_result is not None and len(gen_result) > 0 and gen_result[1] is not None and gen_result[1]:
            job_id = gen_result[1]
            monitor_future = executor.submit(monitor_job, client, job_id)
            monitor_result = None
            
            try:
                c_timeout = (config["requested_seconds"]*600) + 300
                c_timeout = 80000
                if "lora" in config and config["lora"] is not None and len(config["lora"]) != 0:
                    logging.warn("Lora detected, adding some timeout to allow Lora loading")
                    c_timeout = c_timeout + 400
                logging.warn("Using timeout: %s", str(c_timeout))
                monitor_result = monitor_future.result(timeout=c_timeout)
                monitor_future.cancel()
            except (concurrent.futures.TimeoutError, concurrent.futures._base.CancelledError) as e:
                logging.error("Max Execution Time reached")
                logging.error("Stopping current generation")
                result_stop = client.predict(api_name="/end_process_with_update")
                while True:
                    result_current = client.predict(api_name="/check_for_current_job")
                    if result_current is None or len(result_current) == 0 or (len(result_current) > 0 and (result_current[0] is None or result_current[0] == "")):
                        break
                    else:
                        time.sleep(60)
                        result_stop = client.predict(api_name="/end_process_with_update")
                logging.error("Updating skipped param to 2 to database for id: " + str(config["generation_id"]))
                end = time.time()
                config["exec_time_seconds"] = int(end - start)
                config["skipped"] = 2
                config["status"] = 4
                database.update_config(dbms, config)
                raise(e)
            
            if monitor_result is not None and len(monitor_result) > 0 and 'video' in monitor_result[0]:
                generated_video = (os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video']))
                return upscale_and_add_audio(generated_video, config, prompt, start)
    database.delete_wrong_entries(dbms)
    return None, None

def upscale_and_add_audio(generated_video, config, prompt, start):
    config = database.select_config_by_id(dbms, config["generation_id"])
    if generated_video is not None and config["status"] != 4:
        logging.warn("Generation ok")
        config["status"] = 2
        database.update_config(dbms, config)
            
        result_upscale = Client(os.environ.get("FRAMEPACK_ENDPOINT")).predict(
                video_path={"video":handle_file(generated_video)},
                model_key_selected=config["upscale_model"],          
                output_scale_factor_from_slider=2, #2 if config["upscale_model"] == "RealESRGAN_x2plus" else 4,
                tile_size=0,
                enhance_face_ui=True,
                denoise_strength_from_slider=0.5,
                use_streaming=False,
                api_name="/tb_handle_upscale_video"
        )
        config = database.select_config_by_id(dbms, config["generation_id"])
        if len(result_upscale) > 0 and 'video' in result_upscale[0] and config["status"] != 4:
            logging.warn("Upscaling ok")
            file_upscaled = os.environ.get("OUTPUT_PATH") + "postprocessed_output/saved_videos/" + os.path.basename(result_upscale[0]['video'])
            return add_audio(file_upscaled, config, prompt, start)
    database.delete_wrong_entries(dbms)
    return None, None

def add_audio(generated_video, config, prompt, start):
    config["status"] = 3
    database.update_config(dbms, config)
    config = database.select_config_by_id(dbms, config["generation_id"])
    mp4 = add_audio_to_video(generated_video, config, prompt)
    if mp4 is not None:
        logging.warn("Adding audio ok")
        end = time.time()
        config["exec_time_seconds"] = int(end - start)
        config["skipped"] = 1
        config["status"] = 4
        database.update_config(dbms, config)
        logging.warn("Process complete")
        return mp4, database.select_config_by_id(dbms, config["generation_id"])
    database.delete_wrong_entries(dbms)
    return None, None

def generate_image_pre(prompt):
    config = {}
    config["prompt_image"] = None
    config["image_seed"] = random.randint(0, sys.maxsize)
    if prompt is not None and prompt.strip() != "":
        config["prompt_image"] = prompt
    else:
        #reset_workspace()
        message = random.choice(json.loads(os.environ.get('PROMPT_LIST')))
        data = {
            "message": (message),
            "mode": "chat"
        }
        headers = {
            'Authorization': 'Bearer ' + os.environ.get("ANYTHING_LLM_API_KEY")
        }
        anything_llm_url = os.environ.get("ANYTHING_LLM_ENDPOINT") + "/api/v1/workspace/" + os.environ.get("ANYTHING_LLM_WORKSPACE") + "/chat"
        anything_llm_response = requests.post(url=anything_llm_url,
                        data=data,
                        headers=headers)
            
        if (anything_llm_response.status_code == 200):

            story_gen = remove_html_tags_and_content(anything_llm_response.json()["textResponse"].rstrip())
            data_prompt_img = {
                "message": 'Extract one scene from his story, be synthetic, answer with just one sentence: "' + (story_gen) + '"',
                "mode": "chat"
            }
            anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                            data=data_prompt_img,
                            headers=headers)
            if (anything_llm_response_prompt_img.status_code == 200):
                config["prompt_image"] = remove_html_tags_and_content(anything_llm_response_prompt_img.json()["textResponse"].rstrip())
            else:
                raise Exception("Error getting response from AnythingLLM")
        else:
            raise Exception("Error getting response from AnythingLLM")
    gen_image = generate_image(config)
    image = download_file(gen_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"), "png")
    return image

def create_app():
    app = Flask(__name__)
    with app.app_context():
        remove_directory_tree(Path(os.environ.get("OUTPUT_PATH")))
        database.create_db_tables(dbms)
        database.delete_wrong_entries(dbms)
        app.config.from_object(Config())
        app.secret_key = os.environ.get("SECRET_KEY")
        return app

app = create_app()


scheduler = APScheduler()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/minute"],
    storage_uri="memory://",
)


api = Api(app)

nsaivg = api.namespace('aivg', 'AI Video Generator')

@limiter.limit("1/second")
@nsaivg.route('/healthcheck')
class Healthcheck(Resource):
  def get (self):
    return "OK"

@limiter.limit("1/second")
@nsaivg.route('/generate/enhance/')
@nsaivg.route('/generate/enhance/<int:mode>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:use_top>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:use_top>/<int:video_len>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:use_top>/<int:video_len>/<string:message>/')
class GenerateMessage(Resource):
  def post (self, mode = 1, use_top = 0, video_len = 5, message = None):
    final_response = None
    try:
        photo_init = request.files["image"].read() if "image" in request.files else None
        video_init = request.files["video"].read() if "video" in request.files else None
        mp4, config = add_new_generation(video_len, mode, message, None, photo_init, video_init, (True if use_top == 1 else False))
        if mp4 is None:
            
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
        return add_config_response_headers(response, config=config, photo_init=photo_init, video_init=video_init)
    except concurrent.futures.TimeoutError as te:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Video generation took to long', 408)
    except concurrent.futures._base.CancelledError as ce:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Video generation has been cancelled', 410)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Error generating video', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/prompt/<string:prompt>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:use_top>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:use_top>/<int:video_len>/')
class GeneratePrompt(Resource):
  def post (self, prompt = None, mode = 1, use_top = 0, video_len = 5):
    final_response = None
    try:
        photo_init = request.files["image"].read() if "image" in request.files else None
        video_init = request.files["video"].read() if "video" in request.files else None
        mp4, config = add_new_generation(video_len, mode, None, prompt, photo_init, video_init, (True if use_top == 1 else False))
        if mp4 is None:
            
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')        
        return add_config_response_headers(response, config=config, photo_init=photo_init, video_init=video_init)
    except concurrent.futures.TimeoutError as te:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Video generation took to long', 408)
    except concurrent.futures._base.CancelledError as ce:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Video generation has been cancelled', 410)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      
      return make_response('Error generating video', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/image/')
@nsaivg.route('/generate/image/<string:prompt>/')
class GenerateImage(Resource):
  def post (self, prompt = None):
    try:
        if database.select_config_by_skipped(dbms, 0) is None:
            image = generate_image_pre(prompt)
            if image is None:
                return make_response('Error generating image', 500)
            else:
                return send_file(image, attachment_filename=str(uuid.uuid4()) + '.png', mimetype='image/png')
        else:
            return make_response('Another generation in progress', 206)

        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error generating image', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/checkrunning/')
class GenerateImage(Resource):
  def get (self, prompt = None):
    try:
        if database.select_config_by_skipped(dbms, 0) is None:
            return make_response('No generation active', 200)
        else:
            return make_response('Generation in progress', 206)

        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error generating image', 500)


def skipped_and_stop(skipped, generation_id):
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    result_stop = client.predict(api_name="/end_process_with_update")
    while True:
        result_current = client.predict(api_name="/check_for_current_job")
        if result_current is None or len(result_current) == 0 or (len(result_current) > 0 and (result_current[0] is None or result_current[0] == "")):
            config = {}
            config["generation_id"] = generation_id
            config["skipped"] = skipped
            config["status"] = 4
            database.update_config(dbms, config)
            break
        else:
            time.sleep(10)
            result_stop = client.predict(api_name="/end_process_with_update")

@limiter.limit("1/second")
@nsaivg.route('/generate/skipped/<int:skipped>/<int:generation_id>/<int:stop>/')
class GenerateSkipped(Resource):
  def post (self, skipped = None, generation_id = None, stop = 0):
    try:
        config = {}
        config["generation_id"] = generation_id
        config["skipped"] = skipped
        database.update_config(dbms, config)
        if stop == 1:
            Timer(5.0, stop_aivg).start()
        return make_response('Done', 200)        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/top_config/<int:top_config>/<int:generation_id>/')
class GenerateTopConfig(Resource):
  def post (self, top_config = None, generation_id = None):
    try:
        config = database.select_config_by_id(dbms, generation_id)
        config["generation_id"] = generation_id
        config["top_config"] = top_config
        config["skipped"] = 1
        database.update_config(dbms, config)
        return make_response('Done', 200)        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error', 500)

def stop_aivg():
    os.system("pkill -f uwsgi -9")
    #client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    #result_stop = client.predict(api_name="/end_process_with_update")
    #while True:
    #    result_current = client.predict(api_name="/check_for_current_job")
    #    if result_current is None or len(result_current) == 0 or (len(result_current) > 0 and (result_current[0] is None or result_current[0] == "")):
    #        os.system("pkill -f uwsgi -9")
    #    else:
    #        time.sleep(10)
    #        result_stop = client.predict(api_name="/end_process_with_update")

@limiter.limit("1/second")
@nsaivg.route('/stop/')
class Stop(Resource):
  def get (self):
    try:
        Timer(5.0, stop_aivg).start()
        return make_response('Done', 200)        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error', 500)

def get_current_preview_by_extension(extension):
    if len([path for path in Path(os.environ.get("OUTPUT_PATH")+"*").parent.glob('*.'+extension)]) and len([path for path in Path(os.environ.get("OUTPUT_PATH")+"*").parent.glob('*.json')])> 0:
        list_of_ext = glob.glob(os.environ.get("OUTPUT_PATH")+'*.'+extension)
        latest_ext = max(list_of_ext, key=os.path.getctime)
        latest_ext_name = Path(latest_ext).stem
        if extension == "mp4":
            latest_ext_name = "_".join((latest_ext_name).split("_")[:-1])
        list_of_json = glob.glob(os.environ.get("OUTPUT_PATH")+'*.json')
        latest_json = max(list_of_json, key=os.path.getctime)
        latest_json_name = Path(latest_json).stem
        if latest_ext_name == latest_json_name:
            return latest_ext
    return None

def get_status_from_framepack():
    text_ret = ""
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    data = client.predict(api_name="/check_for_current_job") 
    if data is not None and len(data) == 6:       
        if data[4] != '':
            text_ret = text_ret + data[4] + "\n"
        if data[5] != '':
            splitted_data_5 = data[5].split("\n")
            for datah in splitted_data_5:
                if "span" in datah:
                    to_add = datah.strip().replace("<span>","").replace("</span>","")
                    text_ret = text_ret + to_add + "&nbsp;"
    return text_ret.replace("\n","&nbsp;").encode('utf-8').decode('latin-1')

def add_config_response_headers(response, generation_id=None, config=None, photo_init=None, video_init=None):
    if config is None:
        config = database.select_config_by_id(dbms, generation_id)
        response.headers['X-AIVG-Image-Input'] = ("True" if config["has_input_image"] == 1 else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-AIVG-Video-Input'] = ("True" if config["has_input_video"] == 1 else "False").encode('utf-8').decode('latin-1') 
    else:
        response.headers['X-AIVG-Image-Input'] = ("True" if photo_init is not None else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-AIVG-Video-Input'] = ("True" if video_init is not None else "False").encode('utf-8').decode('latin-1') 

    response.headers['X-AIVG-Image-AI-Generated'] = ("True" if config["gen_photo"] == 1 else "False").encode('utf-8').decode('latin-1') 
    response.headers['X-AIVG-Seed'] = str(config["seed"]).encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Image-Seed'] = str(config["image_seed"]).encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Model'] = config["model"].encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Seconds'] = (str(config["requested_seconds"])).encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Window-Size'] = (str(config["window_size"])).encode('utf-8').decode('latin-1') 
    response.headers['X-AIVG-Steps'] = (str(config["steps"])).encode('utf-8').decode('latin-1') 
    response.headers['X-AIVG-Width'] = (str(config["width"])).encode('utf-8').decode('latin-1') 
    response.headers['X-AIVG-Height'] = (str(config["height"])).encode('utf-8').decode('latin-1') 
    if config["distilled_cfg_scale"] is not None:
        response.headers['X-AIVG-Distilled-Cfg-Scale'] = (str(config["distilled_cfg_scale"])).encode('utf-8').decode('latin-1') 
    if config["cfg_scale"] is not None:
        response.headers['X-AIVG-Cfg-Scale'] = (str(config["cfg_scale"])).encode('utf-8').decode('latin-1') 
    if config["cfg_rescale"] is not None:
        response.headers['X-AIVG-Cfg-ReScale'] = (str(config["cfg_rescale"])).encode('utf-8').decode('latin-1') 
    if config["cache_type"] is not None:
        response.headers['X-AIVG-Cache-Type'] = (str(config["cache_type"])).encode('utf-8').decode('latin-1') 
    if str(config["cache_type"]) == "MagCache":
        response.headers['X-AIVG-MagCache-Threshold'] = (str(config["mag_cache_threshold"])).encode('utf-8').decode('latin-1') 
        response.headers['X-AIVG-MagCache-Max-Consecutive-Skips'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        response.headers['X-AIVG-MagCache-Retention-Ratio'] = (str(config["mag_cache_retention_ratio"])).encode('utf-8').decode('latin-1') 
    elif str(config["cache_type"]) == "TeaCache":
        response.headers['X-AIVG-TeaCache-Steps'] = (str(config["tea_cache_steps"])).encode('utf-8').decode('latin-1') 
        response.headers['X-AIVG-TeaCache-Rel-L1-Thresh'] = (str(config["tea_cache_rel_l1_thresh"])).encode('utf-8').decode('latin-1') 
    if "lora" in config and config["lora"] is not None and len(config["lora"]) > 0:
        loras = ','.join(config["lora"])
        response.headers['X-AIVG-Lora'] = loras.encode('utf-8').decode('latin-1')
    #    response.headers['X-AIVG-Lora-Weight'] = str(config["lora_weight"]).encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Upscale-Model'] = config["upscale_model"].encode('utf-8').decode('latin-1')
    #if "prompt" in config and config["prompt"] is not None:
    #    response.headers['X-AIVG-Prompt'] = config["prompt"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
    if "prompt_image" in config and config["prompt_image"] is not None:
        response.headers['X-AIVG-Prompt-Image'] = config["prompt_image"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
    if 'exec_time_seconds' in config and config['exec_time_seconds'] is not None and config['exec_time_seconds'] != 0:
        response.headers['X-AIVG-Execution-Time'] = str(timedelta(seconds=int(config['exec_time_seconds']))).encode('utf-8').decode('latin-1')
        response.headers['X-AIVG-Sec-Per-GenSec'] = str(int(config['exec_time_seconds'])/int(config["requested_seconds"])).encode('utf-8').decode('latin-1')
    response.headers['X-AIVG-Generation-Id'] = str(config['generation_id']).encode('utf-8').decode('latin-1')
    return response

@limiter.limit("1/second")
@nsaivg.route('/generate/check/job/')
class GenerateCheck(Resource):
  def post (self):
    try:
        value = database.select_config_by_skipped(dbms, 0)
        generation_id = value[0] if value != None and len(value) == 4 else None
        status = value[2] if value != None and len(value) == 4 else None
        if generation_id is not None and status is not None:
            if status == 0:
                return add_config_response_headers(make_response('Starting', 201), generation_id=int(generation_id))
            elif status == 1:
                mp4 = get_current_preview_by_extension("mp4")
                png = get_current_preview_by_extension("png") if mp4 is None else None
                if mp4 is not None or png is not None:
                    response = send_file((png if mp4 is None else mp4), attachment_filename=str(uuid.uuid4()) + (".png" if mp4 is None else ".mp4"), mimetype=('image/png' if mp4 is None else 'video/mp4'))
                    response = add_config_response_headers(response, generation_id=int(generation_id))
                    response.headers['X-AIVG-File-Name'] = str(os.path.basename(png if mp4 is None else mp4)).encode('utf-8').decode('latin-1')
                    if value[3] == "framepack":
                        response.headers['X-AIVG-Check-Current-Job'] = get_status_from_framepack().encode('utf-8').decode('latin-1')
                    return response
                else:
                    response = make_response('Starting', 205)
                    response = add_config_response_headers(response, generation_id=int(generation_id))
                    if value[3] == "framepack":
                        response.headers['X-AIVG-Check-Current-Job'] = get_status_from_framepack().encode('utf-8').decode('latin-1') 
                    return response
            elif status == 2:
                return add_config_response_headers(make_response('Upscaling', 202), generation_id=int(generation_id))
            elif status == 3:
                return add_config_response_headers(make_response('Adding audio', 204), generation_id=int(generation_id))
        return make_response('No jobs running', 206)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error', 500)

#@scheduler.task('interval', id='generate_loop', hours = 6)
#def generate_loop():
#    add_new_generation(random.randint(15,60))

limiter.init_app(app)
scheduler.init_app(app)
scheduler.start()

if __name__ == '__main__':
  app.run()
