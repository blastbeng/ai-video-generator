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
import queue
import multiprocessing
import database

from functools import wraps
from concurrent import futures
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask
from flask import send_file
from flask import Response
from flask import make_response
from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api
from flask_restx import Resource
from flask_restx import reqparse
from os.path import dirname
from os.path import join
from pathlib import Path
from threading import Thread
from flask_apscheduler import APScheduler
from gradio_client import Client, handle_file

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)



logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')
        
log = logging.getLogger('werkzeug')
log.setLevel(int(os.environ.get("LOG_LEVEL")))

dbms = database.Database(database.SQLITE, dbname='configs.sqlite3')

class ThreadPoolExecutorWithQueueSizeLimit(futures.ThreadPoolExecutor):
    def __init__(self, maxsize=10, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)

executor = ThreadPoolExecutorWithQueueSizeLimit(max_workers=1) 

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return
    send_end.send(result)

@parametrized
def run_with_timer(func, max_execution_time):
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func
        
        ## PART 2
        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("Exceeded Execution Time")
        result = recv_end.recv()

        if isinstance(result, Exception):
            raise result

        return result

    return wrapper

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

def generate_image(prompt):
    result = text2img({
        "prompt": prompt,
        "negative_prompt": os.environ.get("NEGATIVE_PROMPT"),
        "performance_selection": "Quality",
        "aspect_ratios_selection": "704*1344",
        "guidance_scale": 20.0,
        "image_number": 1,
        "image_seed": random.randint(0, 9223372036854775807),
        "async_process": False,
        "style_selections": [ "Fooocus V2", "Fooocus V2 (Optional)", "Fooocus Enhance", "Fooocus Sharp", "Fooocus Negative", "Fooocus Cinematic", "Cinematic Diva", "Fooocus Photograph" ]
        })
    if len(result) > 0 and 'url' in result[0]:
        return result[0]['url']
    else:
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

def add_audio_to_video(file_path, video_len):
    url = os.environ.get("MMAUDIO_ENDPOINT") + "/process"
    payload = {
        #'prompt': prompt, 
        'negative_prompt': "", 
        'variant': "large_44k_v2", 
        'duration': str(video_len)
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


def add_new_generation_framepack(video_len, mode, gen_photo, message, prompt, image, video):
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    result = client.predict(
		api_name="/check_for_current_job"
    )
    if result is not None and len(result) > 0 and (result[0] is None or result[0] == ""):
        
        prompt_image = None
        if prompt is None:
            message = random.choice(json.loads(os.environ.get('PROMPT_LIST'))) if message is None else message
            prompt_image = message
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
                
                prompt = anything_llm_response.json()["textResponse"].rstrip()
                if gen_photo:
                    data_prompt_img = {
                        "message": 'Extract one scene this story, be synthetic, answer with just one sentence: "' + prompt + '"',
                        "mode": "chat"
                    }
                    anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                                    data=data_prompt_img,
                                    headers=headers)
                    if (anything_llm_response_prompt_img.status_code == 200):
                        prompt_image = anything_llm_response_prompt_img.json()["textResponse"].rstrip()
                    else:
                        raise Exception("Error getting response from AnythingLLM")
                else:
                    prompt_image = prompt
            else:
                raise Exception("Error getting response from AnythingLLM")
        else:
            prompt_image = prompt
        photo_init = None
        video_init = None
        if image is not None:
            photo_init = save_file(image, ".png")
        elif video is not None:
            video_init = save_file(video, ".mp4")
        elif gen_photo:
            start_image = generate_image(prompt_image)
            photo_init = download_file(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"), "png")

        photo_end = None
        loras = []
        #loras = [
        #    "DigitalWave",
        #    "dolly-zoom-hunyuan-v1.0-vfx_ai",
        #    "CandidChange v2 e45",
        #    "Sernia_Iori_Flameheartセルニア伊織フレイムハートれでぃばと",
        #    "parkour-freerunning-hunyuan-v1.0-vfx_ai",
        #    "HunyuanVideo - Glitchy DV Cam - Trigger is yellowjackets intro filmstyle",
        #    "idle_dance",
        #    "DreamPunk_e33",
        #    "anaglyph 3D",
        #    "executive_order_40_epochs",
        #    "aifantasia",
        #    "hunyuan_80s_fantasyv1_5_comfy",
        #    "Retro_Styles_Hunyuan",
        #    "sidebyside_E75",
        #    "1950s_epoch50",
        #    "defaultDance_preview",
        #    "Anne-IL-10",
        #    "pubg_146_framepack",
        #    "cinematron",
        #    "adapter_model",
        #    "Odyssey_Space_Suit",
        #    "fluidart-v1_hunyuanvideo_e28",
        #    "frostyfaces",
        #    "hunyuan_ancientrome",
        #    "s40r1k1d0",
        #    "hunyuan_80s_horror_1000",
        #    "lone-cyclist-pruned",
        #    "Tw3rk_e15",
        #    "Slut-000009",
        #    "t5xxl_fp8_e4m3fn",
        #    "ladyjaye_Il",
        #    "xjx-TokyoRacerV2-comfy",
        #    "Shiraishi_Ken",
        #    "boxing_epoch20",
        #    "pixar_7_epochs",
        #    "Sexy_Dance_e15",
        #    "Wedding_Dress",
        #    "hunyuan_kungfu_600",
        #    "Neon_Punk_hyv",
        #    "dji_20250103_02-47-51_epoch9",
        #    "360c4m3r4_e100_only_double_blocks",
        #    "Digital_Human_Hunyuan",
        #    "ph2t-h0n-v1.0",
        #    "cinematik_flux50epoch",
        #    "RFX.XT404.V0.0.1",
        #    "thanos-snap-r512-768-e20",
        #    "cywo1_Cyber_Woman_n1",
        #    "Graphical_Clothes_hyv",
        #    "HeavyMetal512Epoch65",
        #    "adapter_modelsq",
        #    "closeupface-v1.1",
        #    "Dom_and_Sub",
        #    "sd40_converted",
        #    "matrix-bullet-time-hunyuan-v1.0-vfx_ai",
        #    "hunyuan_mtv_grind_500",
        #    "venom_hunyuan_video_v1_e40",
        #    "hunyuan_darkraw",
        #    "poplock10",
        #    "cyberp@nk",
        #    "GTA_epoch8",
        #    "cat_epoch20",
        #    "nsbCheckpoint_skb",
        #    "kxsr_walking_anim_v1-5",
        #    "fxf-tokyoMeet-comfy",
        #    "Comic_Art_Illustration_Style.fp1600018",
        #    "high-speed-drone-shot-hunyuan-v1.0-heavy-vfx_ai",
        #    "混元-星空6",
        #    "animal_documentary_epoch20",
        #    "framepack_dabaichui",
        #]
        mp4, config = get_video(client, mode, photo_init, video_init, prompt, loras, video_len)
        return mp4, config
    else:
        return False, None
    return None, None

def get_config(mode, photo_init, video_init, requested_seconds, prompt):
    config = {}
    model = "F1" if mode else "Original"
    if video_init is not None:
        model = "Video F1" if mode else "Video"
    config["model"] = model
    config["has_input_image"] = 0 if photo_init is None else 1
    config["has_input_video"] = 0 if video_init is None else 1
    config["requested_seconds"] = requested_seconds
    config["seed"] = random.randint(0, 9223372036854775807)
    config["window_size"] = random.randint(9, 15)
    config["steps"] = random.randint(25, 50)
    config["cache_type"] = random.choice(["MagCache","MagCache"])
    config["tea_cache_steps"] = random.randint(1, 50)
    config["tea_cache_rel_l1_thresh"] = round(random.uniform(0, 1.01), 2)
    config["mag_cache_threshold"] = round(random.uniform(0, 1), 2)
    config["mag_cache_max_consecutive_skips"] = random.randint(1, 5)
    config["mag_cache_retention_ratio"] = round(random.uniform(-0.01, 1), 2)
    config["distilled_cfg_scale"] = round(random.uniform(0.9, 32.1), 1)
    config["cfg_scale"] = round(random.uniform(0.9, 3.1), 1)
    config["cfg_rescale"] = round(random.uniform(-0.01, 1.01), 2)
    config["prompt"] = prompt
    return config

def start_video_gen(client, config, photo_init, video_init, loras):
    result = client.predict(
        selected_model=config["model"],
        param_1=handle_file(photo_init) if photo_init is not None else None,
        param_2=({"video":handle_file(video_init)}) if video_init is not None else None,
        param_3=None,
        param_4=1,
        param_5=config["prompt"],
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
        param_24=loras, #param_24=random.sample(loras, random.randint(1, len(loras))),
        param_25=512, #param_25=512,
        param_26=768, #param_26=768,
        param_27=True,
        param_28=5,
        #param_30=round(random.uniform(-0.01, 2.01), 2),
        #param_31=round(random.uniform(-0.01, 2.01), 2),
        #param_32=round(random.uniform(-0.01, 2.01), 2),
        #param_33=round(random.uniform(-0.01, 2.01), 2),
        #param_34=round(random.uniform(-0.01, 2.01), 2),
        #param_35=round(random.uniform(-0.01, 2.01), 2),
        #param_36=round(random.uniform(-0.01, 2.01), 2),
        #param_37=round(random.uniform(-0.01, 2.01), 2),
        #param_38=round(random.uniform(-0.01, 2.01), 2),
        #param_39=round(random.uniform(-0.01, 2.01), 2),
        #param_40=round(random.uniform(-0.01, 2.01), 2),
        #param_41=round(random.uniform(-0.01, 2.01), 2),
        #param_42=round(random.uniform(-0.01, 2.01), 2),
        #param_43=round(random.uniform(-0.01, 2.01), 2),
        #param_44=round(random.uniform(-0.01, 2.01), 2),
        #param_45=round(random.uniform(-0.01, 2.01), 2),
        #param_46=round(random.uniform(-0.01, 2.01), 2),
        #param_47=round(random.uniform(-0.01, 2.01), 2),
        #param_48=round(random.uniform(-0.01, 2.01), 2),
        #param_49=round(random.uniform(-0.01, 2.01), 2),
        #param_50=round(random.uniform(-0.01, 2.01), 2),
        #param_51=round(random.uniform(-0.01, 2.01), 2),
        #param_52=round(random.uniform(-0.01, 2.01), 2),
        #param_53=round(random.uniform(-0.01, 2.01), 2),
        #param_54=round(random.uniform(-0.01, 2.01), 2),
        #param_55=round(random.uniform(-0.01, 2.01), 2),
        #param_56=round(random.uniform(-0.01, 2.01), 2),
        #param_57=round(random.uniform(-0.01, 2.01), 2),
        #param_58=round(random.uniform(-0.01, 2.01), 2),
        #param_59=round(random.uniform(-0.01, 2.01), 2),
        #param_60=round(random.uniform(-0.01, 2.01), 2),
        #param_61=round(random.uniform(-0.01, 2.01), 2),
        #param_62=round(random.uniform(-0.01, 2.01), 2),
        #param_63=round(random.uniform(-0.01, 2.01), 2),
        #param_64=round(random.uniform(-0.01, 2.01), 2),
        #param_65=round(random.uniform(-0.01, 2.01), 2),
        #param_66=round(random.uniform(-0.01, 2.01), 2),
        #param_67=round(random.uniform(-0.01, 2.01), 2),
        #param_68=round(random.uniform(-0.01, 2.01), 2),
        #param_69=round(random.uniform(-0.01, 2.01), 2),
        #param_70=round(random.uniform(-0.01, 2.01), 2),
        #param_71=round(random.uniform(-0.01, 2.01), 2),
        #param_72=round(random.uniform(-0.01, 2.01), 2),
        #param_73=round(random.uniform(-0.01, 2.01), 2),
        #param_74=round(random.uniform(-0.01, 2.01), 2),
        #param_75=round(random.uniform(-0.01, 2.01), 2),
        #param_76=round(random.uniform(-0.01, 2.01), 2),
        #param_77=round(random.uniform(-0.01, 2.01), 2),
        #param_78=round(random.uniform(-0.01, 2.01), 2),
        #param_79=round(random.uniform(-0.01, 2.01), 2),
        #param_80=round(random.uniform(-0.01, 2.01), 2),
        #param_81=round(random.uniform(-0.01, 2.01), 2),
        #param_82=round(random.uniform(-0.01, 2.01), 2),
        #param_83=round(random.uniform(-0.01, 2.01), 2),
        #param_84=round(random.uniform(-0.01, 2.01), 2),
        #param_85=round(random.uniform(-0.01, 2.01), 2),
        #param_86=round(random.uniform(-0.01, 2.01), 2),
        #param_87=round(random.uniform(-0.01, 2.01), 2),
        #param_88=round(random.uniform(-0.01, 2.01), 2),
        #param_89=round(random.uniform(-0.01, 2.01), 2),
        #param_90=round(random.uniform(-0.01, 2.01), 2),
        #param_91=round(random.uniform(-0.01, 2.01), 2),
        #param_92=round(random.uniform(-0.01, 2.01), 2),
        #param_93=round(random.uniform(-0.01, 2.01), 2),
        #param_94=round(random.uniform(-0.01, 2.01), 2),
        #param_95=round(random.uniform(-0.01, 2.01), 2),
        #param_96=round(random.uniform(-0.01, 2.01), 2),
        #param_97=round(random.uniform(-0.01, 2.01), 2),
        #param_98=round(random.uniform(-0.01, 2.01), 2),
        api_name="/handle_start_button"
    )
    return result

@run_with_timer(max_execution_time=10800)
def monitor_job(job_id):
    monitor_result = client.predict(
            job_id=job_id,
            api_name="/monitor_job"
    )
    return monitor_result


def get_video(client, mode, photo_init, video_init, prompt, loras, requested_seconds):

    skipped = None

    for n in range(900):
        config = get_config(mode, photo_init, video_init, requested_seconds, prompt)
        value = database.select_config(dbms, config)
        if skipped is not None and skipped == 1:
            logging.warn("Found skipped params: %s", str(config))
        else:
            break

    if skipped is None or skipped == 0 or skipped == 2:
        if skipped is None:
            logging.warn("Saving params to database")
            database.insert_wrong_config(dbms, config)
        logging.warn("Launching with params: %s", str(config))

        gen_result = None
        try:
            gen_result = start_video_gen(client, config, photo_init, video_init, loras)
            if gen_result is not None and len(gen_result) > 0 and gen_result[1] is not None and gen_result[1] != "":
                job_id = gen_result[1]
                monitor_result = monitor_job(job_id)
                if monitor_result is not None and len(monitor_result) > 0 and 'video' in monitor_result[0]:
                    generated_video = (os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video']))
                    if generated_video is not None:
                        logging.warn("Generation ok")
                            
                        if skipped is None or skipped == 0:
                            logging.warn("Updating skipped param to 0 to database for config: " + str(config))
                            database.update_ok_config(dbms, config)
                        result_upscale = client.predict(
                                video_path={"video":handle_file(generated_video)},
                                model_key_selected="RealESRGAN_x2plus",
                                output_scale_factor_from_slider=2,
                                tile_size=0,
                                enhance_face_ui=True,
                                denoise_strength_from_slider=0.5,
                                use_streaming=False,
                                api_name="/tb_handle_upscale_video"
                        )
                        if len(result_upscale) > 0 and 'video' in result_upscale[0]:
                            logging.warn("Upscaling ok")
                            file_upscaled = os.environ.get("OUTPUT_PATH") + "postprocessed_output/saved_videos/" + os.path.basename(result_upscale[0]['video'])
                            mp4 = add_audio_to_video(file_upscaled, config["requested_seconds"])
                            if mp4 is not None:
                                logging.warn("Adding audio ok")
                                
                                logging.warn("Process complete")
                                return mp4, config
                return None, None
        except TimeoutError:
            logging.error("Updating skipped param to 1 to database for config: " + str(config))
            database.update_skipped_config(dbms, config)
            return None, None
    else:
        logging.error("I haven't found any working config")
    return None, None

def generate_image_pre(prompt):
    prompt_image = None
    if prompt is not None and prompt.strip() != "":
        prompt_image = prompt
    else:
        message = random.choice(json.loads(os.environ.get('PROMPT_LIST')))
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

            story_gen = anything_llm_response.json()["textResponse"].rstrip()
            data_prompt_img = {
                "message": 'Extract one scene this story, be synthetic, answer with just one sentence: "' + story_gen + '"',
                "mode": "chat"
            }
            anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                            data=data_prompt_img,
                            headers=headers)
            if (anything_llm_response_prompt_img.status_code == 200):
                prompt_image = anything_llm_response_prompt_img.json()["textResponse"].rstrip()
            else:
                raise Exception("Error getting response from AnythingLLM")
        else:
            raise Exception("Error getting response from AnythingLLM")
        
    gen_image = generate_image(prompt_image)
    image = download_file(gen_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"), "png")
    return image

def create_app():
    app = Flask(__name__)
    with app.app_context():
        remove_directory_tree(Path(os.environ.get("OUTPUT_PATH")))
        database.create_db_tables(dbms)
        database.delete_wrong_entries(dbms)
        return app

app = create_app()
class Config:    
    SCHEDULER_API_ENABLED = True

scheduler = APScheduler()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/minute"],
    storage_uri="memory://",
)

app.config.from_object(Config())
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
@nsaivg.route('/generate/enhance/<int:mode>/<int:gen_photo>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:gen_photo>/<int:video_len>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:gen_photo>/<int:video_len>/<string:message>/')
class GenerateMessage(Resource):
  def post (self, mode = 1, gen_photo = 1, video_len = 5, message = None):
    try:
        mp4, config = add_new_generation_framepack(video_len, (True if mode == 1 else False), (True if gen_photo == 1 else False), message, None, request.files["image"].read() if "image" in request.files else None, request.files["video"].read() if "video" in request.files else None)
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
        response.headers['X-FramePack-Seed'] = str(config["seed"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Model'] = config["model"].encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Loras'] =  (', '.join(loras)).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Seconds'] = (str(config["requested_seconds"])).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Has-Image-Input'] = ("True" if photo_init is not None else False).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Has-Video-Input'] = ("True" if video_init is not None else False).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Window-Size'] = (str(config["window_size"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Steps'] = (str(config["steps"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-DistilledCfgScale'] = (str(config["distilled_cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-CfgScale'] = (str(config["cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-CfgReScale'] = (str(config["cfg_rescale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cache-Tye'] = (str(config["cache_type"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Threshold'] = (str(config["mag_cache_threshold"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Max-Consecutive-Skips'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Retention-Ratio'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn'] = str(hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-HunyuanVideo_dashtoon_keyframe_lora_converted_comfy_bf16'] = str(HunyuanVideo_dashtoon_keyframe_lora_converted_comfy_bf16_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvid_I2V_lora_hair_growth'] = str(hyvid_I2V_lora_hair_growth_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvideo_FastVideo_LoRA-fp8'] = str(hyvideo_FastVideo_LoRA_fp8_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvid_I2V_lora_embrace'] = str(hyvid_I2V_lora_embrace_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-HunyuanVideo_dashtoon_keyframe_lora_converted_bf16'] = str(HunyuanVideo_dashtoon_keyframe_lora_converted_bf16_weight).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Prompt'] = prompt.encode('utf-8').decode('latin-1')
        return response
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error generating video', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/prompt/<string:prompt>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:gen_photo>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:gen_photo>/<int:video_len>/')
class GeneratePrompt(Resource):
  def post (self, prompt = None, mode = 1, gen_photo = 1, video_len = 5):
    try:
        mp4, config = add_new_generation_framepack(video_len, (True if mode == 1 else False), (True if gen_photo == 1 else False), None, prompt, request.files["image"].read() if "image" in request.files else None, request.files["video"].read() if "video" in request.files else None)
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
        response.headers['X-FramePack-Seed'] = str(config["seed"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Model'] = config["model"].encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Loras'] =  (', '.join(loras)).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Seconds'] = (str(config["requested_seconds"])).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Has-Image-Input'] = ("True" if photo_init is not None else False).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Has-Video-Input'] = ("True" if video_init is not None else False).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Window-Size'] = (str(config["window_size"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Steps'] = (str(config["steps"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-DistilledCfgScale'] = (str(config["distilled_cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-CfgScale'] = (str(config["cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-CfgReScale'] = (str(config["cfg_rescale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cache-Tye'] = (str(config["cache_type"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Threshold'] = (str(config["mag_cache_threshold"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Max-Consecutive-Skips'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-MagCache-Retention-Ratio'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn'] = str(hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-HunyuanVideo_dashtoon_keyframe_lora_converted_comfy_bf16'] = str(HunyuanVideo_dashtoon_keyframe_lora_converted_comfy_bf16_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvid_I2V_lora_hair_growth'] = str(hyvid_I2V_lora_hair_growth_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvideo_FastVideo_LoRA-fp8'] = str(hyvideo_FastVideo_LoRA_fp8_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-hyvid_I2V_lora_embrace'] = str(hyvid_I2V_lora_embrace_weight).encode('utf-8').decode('latin-1') 
        #response.headers['X-FramePack-HunyuanVideo_dashtoon_keyframe_lora_converted_bf16'] = str(HunyuanVideo_dashtoon_keyframe_lora_converted_bf16_weight).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Prompt'] = prompt.encode('utf-8').decode('latin-1')
        return response
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
        future = executor.submit(generate_image_pre, prompt)
        image = future.result()
        if image is None:
            return make_response('Error generating image', 500)
        else:
            return send_file(image, attachment_filename=str(uuid.uuid4()) + '.png', mimetype='image/png')
        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error generating image', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/stop/')
class GeneratePrompt(Resource):
  def get (self):
    try:
        client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
        result = client.predict(
                api_name="/end_process_with_update"
        )
        logging.info("%s", str(result))
        return make_response('Stopping current generation...', 200)
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
