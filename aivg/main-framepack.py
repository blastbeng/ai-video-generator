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
from threading import Thread
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
        "prompt": prompt.replace("\n", " "),
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

def add_audio_to_video(file_path, config):
    url = os.environ.get("MMAUDIO_ENDPOINT") + "/process"
    payload = {
        #'prompt': config["prompt"], 
        'negative_prompt': "music", 
        'variant': "large_44k_v2", 
        'duration': str(config["requested_seconds"]), 
        'seed': str(config["seed"]), 
        'full_precision': True, 
        'seed': str(config["seed"])
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

def add_new_generation_framepack(video_len, mode, message, prompt, image, video):
    time.sleep(5)
    if database.select_config_by_skipped(dbms, 0) is None:

        for n in range(10000):
            config = get_config(mode, image, video, video_len)
            value = database.select_config(dbms, config)
            if value is not None:
                config["generation_id"] = value[0]
                config["skipped"] =  value[1]
            if config["skipped"] is not None and config["skipped"] == 2:
                logging.warn("Found skipped params: %s", str(config))
            else:
                break

        if config["skipped"] is None or config["skipped"] == 0 or config["skipped"] == 1:
            
            if config["skipped"] is None:
                logging.warn("Saving params to database")
                config["generation_id"] = database.insert_wrong_config(dbms, config)
                config["skipped"] = 0
            elif config["skipped"] is not None and config["skipped"] == "1":
                config["skipped"] = 0
                database.update_config(dbms, config)
        
            config["prompt_image"] = None
            if prompt is None:
                #reset_workspace()
                message = (str(random.choice(json.loads(os.environ.get('PROMPT_LIST'))) if message is None else message))
                config["prompt_image"] = message
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
                    
                    prompt = remove_html_tags_and_content(anything_llm_response.json()["textResponse"].rstrip())
                    if config["gen_photo"] == 1:
                        data_prompt_img = {
                            "message": 'Extract one scene this story, be synthetic, answer with just one sentence: "' + (str(prompt)) + '"',
                            "mode": "chat"
                        }
                        anything_llm_response_prompt_img = requests.post(url=anything_llm_url,
                                        data=data_prompt_img,
                                        headers=headers)
                        if (anything_llm_response_prompt_img.status_code == 200):
                            config["prompt_image"] = remove_html_tags_and_content(anything_llm_response_prompt_img.json()["textResponse"].rstrip())
                        else:
                            database.delete_wrong_entries(dbms)
                            raise Exception("Error getting response from AnythingLLM")
                    else:
                        config["prompt_image"] = prompt
                else:
                    database.delete_wrong_entries(dbms)
                    raise Exception("Error getting response from AnythingLLM")
            else:
                config["prompt_image"] = prompt
            photo_init = None
            video_init = None
            if image is not None:
                photo_init = save_file(image, ".png")
            elif video is not None:
                video_init = save_file(video, ".mp4")
            elif config["gen_photo"] == 1:
                start_image = generate_image(config["prompt_image"])
                photo_init = download_file(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"), "png")

            config["prompt"] = prompt

            photo_end = None
            mp4, config = get_video(mode, photo_init, video_init, config)
            return mp4, config
        else:
            logging.error("I haven't found any working config")
    else:
        return False, None
    return None, None

def get_config(mode, image, video, requested_seconds):
    config = {}
    gen_photo = 1 #random.randint(0, 1)
    model = "F1" if mode else "Original"
    if video is not None:
        model = "Video F1" if mode else "Video"
    config["model"] = model
    config["has_input_image"] = 1 if image is not None or gen_photo == 1 else 0
    config["has_input_video"] = 1 if video is not None else 0
    config["requested_seconds"] = requested_seconds
    config["seed"] = random.randint(0, 9223372036854775807)
    config["window_size"] = random.randint(9, 15)
    config["steps"] = random.randint(10, 50)
    config["cache_type"] = random.choice(["MagCache","TeaCache"])
    config["tea_cache_steps"] = random.randint(1, 50) if config["cache_type"] == "TeaCache" else None
    config["tea_cache_rel_l1_thresh"] = round(random.uniform(0.01, 1), 2) if config["cache_type"] == "TeaCache" else None
    config["mag_cache_threshold"] = round(random.uniform(0.01, 1), 2) if config["cache_type"] == "MagCache" else None
    config["mag_cache_max_consecutive_skips"] = random.randint(1, 5) if config["cache_type"] == "MagCache" else None
    config["mag_cache_retention_ratio"] = round(random.uniform(0, 1), 2) if config["cache_type"] == "MagCache" else None
    config["distilled_cfg_scale"] = round(random.uniform(1.0, 32), 1)
    config["cfg_scale"] = round(random.uniform(1, 3), 1)
    config["cfg_rescale"] = round(random.uniform(0, 1), 2)
    #config["lora"] = ["hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn"]
    #config["lora"] = ["hyvideo_FastVideo_LoRA-fp8"] if (bool(random.getrandbits(1))) else [] #["hyvideo_FastVideo_LoRA-fp8"]
    config["lora"] = None
    config["lora_weight"] = round(random.uniform(0, 2), 2) if config["lora"] is not None and len(config["lora"]) > 0 else 0
    config["prompt"] = ""
    config["gen_photo"] = gen_photo
    config["skipped"] = None
    config['exec_time_seconds'] = 0
    return config

def start_video_gen(client, config, photo_init, video_init):
    result = client.predict(
        selected_model=config["model"],
        param_1=handle_file(photo_init) if photo_init is not None else None,
        param_2=({"video":handle_file(video_init)}) if video_init is not None else None,
        param_3=None,
        param_4=1,
        param_5=config["prompt"].replace("\n", " "),
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
        param_24=config["lora"] if config["lora"] is not None and len(config["lora"]) > 0 else [],
        param_25=512, #param_25=512,
        param_26=768, #param_26=768,
        param_27=True,
        param_28=5,
		param_30=config["lora_weight"],
        api_name="/handle_start_button"
    )
    return result

def monitor_job(client, job_id):
    monitor_result = client.predict(
            job_id=job_id,
            api_name="/monitor_job"
    )
    return monitor_result


def get_video(mode, photo_init, video_init, config):
    start = time.time()
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    logging.warn("Launching with params: %s", str(config))
    gen_result = start_video_gen(client, config, photo_init, video_init)
    if gen_result is not None and len(gen_result) > 0 and gen_result[1] is not None and gen_result[1] != "":
        job_id = gen_result[1]
        monitor_future = executor.submit(monitor_job, client, job_id)
        monitor_result = None
        
        try:
            c_timeout = (config["requested_seconds"]*200) + 300
            if config["lora"] is not None and len(config["lora"]) != 0:
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
            database.update_config(dbms, config)
            raise(e)
        
        if monitor_result is not None and len(monitor_result) > 0 and 'video' in monitor_result[0]:
            generated_video = (os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video']))
            if generated_video is not None:
                logging.warn("Generation ok")
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
                    mp4 = add_audio_to_video(file_upscaled, config)
                    if mp4 is not None:
                        logging.warn("Adding audio ok")
                        end = time.time()
                        config["exec_time_seconds"] = int(end - start)
                        config["skipped"] = 1
                        database.update_config(dbms, config)
                        logging.warn("Process complete")
                        return mp4, config
    database.delete_wrong_entries(dbms)
    return None, None

def generate_image_pre(prompt):
    prompt_image = None
    if prompt is not None and prompt.strip() != "":
        prompt_image = prompt
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
                prompt_image = remove_html_tags_and_content(anything_llm_response_prompt_img.json()["textResponse"].rstrip())
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
@nsaivg.route('/generate/enhance/<int:mode>/<int:video_len>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:video_len>/<string:message>/')
class GenerateMessage(Resource):
  def post (self, mode = 1, video_len = 5, message = None):
    final_response = None
    try:
        photo_init = request.files["image"].read() if "image" in request.files else None
        video_init = request.files["video"].read() if "video" in request.files else None
        mp4, config = add_new_generation_framepack(video_len, (True if mode == 1 else False), message, None, photo_init, video_init)
        if mp4 is None:
            
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
        response.headers['X-FramePack-Image-Input'] = ("True" if photo_init is not None else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Image-AI-Generated'] = ("True" if config["gen_photo"] == 1 else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Video-Input'] = ("True" if video_init is not None else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Seed'] = str(config["seed"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Model'] = config["model"].encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Seconds'] = (str(config["requested_seconds"])).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Window-Size'] = (str(config["window_size"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Steps'] = (str(config["steps"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Distilled-CfgS-cale'] = (str(config["distilled_cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cfg-Scale'] = (str(config["cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cfg-ReScale'] = (str(config["cfg_rescale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cache-Type'] = (str(config["cache_type"])).encode('utf-8').decode('latin-1') 
        if str(config["cache_type"]) == "MagCache":
            response.headers['X-FramePack-MagCache-Threshold'] = (str(config["mag_cache_threshold"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-MagCache-Max-Consecutive-Skips'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-MagCache-Retention-Ratio'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        elif  str(config["cache_type"]) == "TeaCache":
            response.headers['X-FramePack-TeaCache-Steps'] = (str(config["tea_cache_steps"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-TeaCache-Rel-L1-Thresh'] = (str(config["tea_cache_rel_l1_thresh"])).encode('utf-8').decode('latin-1') 
        if config["lora"] is not None and len(config["lora"]) > 0:
            response.headers['X-FramePack-Lora'] = (', '.join(config["lora"])).encode('utf-8').decode('latin-1')
            response.headers['X-FramePack-Lora-Weight'] = str(config["lora_weight"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Prompt'] = config["prompt"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
        if config["prompt_image"] is not None:
            response.headers['X-FramePack-Prompt-Image'] = config["prompt_image"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Execution-Time'] = (str(config['exec_time_seconds']) + " seconds").encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Generation-Id'] = str(config['generation_id']).encode('utf-8').decode('latin-1')
        
        return response
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
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:video_len>/')
class GeneratePrompt(Resource):
  def post (self, prompt = None, mode = 1, video_len = 5):
    final_response = None
    try:
        photo_init = request.files["image"].read() if "image" in request.files else None
        video_init = request.files["video"].read() if "video" in request.files else None
        mp4, config = add_new_generation_framepack(video_len, (True if mode == 1 else False), None, prompt, photo_init, video_init)
        if mp4 is None:
            
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        
        response = send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
        response.headers['X-FramePack-Image-Input'] = ("True" if photo_init is not None else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Image-AI-Generated'] = ("True" if config["gen_photo"] == 1 else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Video-Input'] = ("True" if video_init is not None else "False").encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Seed'] = str(config["seed"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Model'] = config["model"].encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Seconds'] = (str(config["requested_seconds"])).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Window-Size'] = (str(config["window_size"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Steps'] = (str(config["steps"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Distilled-Cfg-Scale'] = (str(config["distilled_cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cfg-Scale'] = (str(config["cfg_scale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cfg-ReScale'] = (str(config["cfg_rescale"])).encode('utf-8').decode('latin-1') 
        response.headers['X-FramePack-Cache-Type'] = (str(config["cache_type"])).encode('utf-8').decode('latin-1') 
        if str(config["cache_type"]) == "MagCache":
            response.headers['X-FramePack-MagCache-Threshold'] = (str(config["mag_cache_threshold"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-MagCache-Max-Consecutive-Skips'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-MagCache-Retention-Ratio'] = (str(config["mag_cache_max_consecutive_skips"])).encode('utf-8').decode('latin-1') 
        elif  str(config["cache_type"]) == "TeaCache":
            response.headers['X-FramePack-TeaCache-Steps'] = (str(config["tea_cache_steps"])).encode('utf-8').decode('latin-1') 
            response.headers['X-FramePack-TeaCache-Rel-L1-Thresh'] = (str(config["tea_cache_rel_l1_thresh"])).encode('utf-8').decode('latin-1') 
        if config["lora"] is not None and len(config["lora"]) > 0:
            response.headers['X-FramePack-Lora'] = (', '.join(config["lora"])).encode('utf-8').decode('latin-1')
            response.headers['X-FramePack-Lora-Weight'] = str(config["lora_weight"]).encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Prompt'] = config["prompt"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
        if config["prompt_image"] is not None:
            response.headers['X-FramePack-Prompt-Image'] = config["prompt_image"].replace("\n","&nbsp;").encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Execution-Time'] = (str(config['exec_time_seconds']) + " seconds").encode('utf-8').decode('latin-1')
        response.headers['X-FramePack-Generation-Id'] = str(config['generation_id']).encode('utf-8').decode('latin-1')
        
        return response
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

@limiter.limit("1/second")
@nsaivg.route('/generate/skipped/<int:skipped>/<int:generation_id>/')
class GenerateSkipped(Resource):
  def post (self, skipped = None, generation_id = None):
    try:
        config = {}
        config["generation_id"] = generation_id
        config["skipped"] = skipped
        database.update_config(dbms, config)
        return make_response('Done', 200)        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return c

@limiter.limit("1/second")
@nsaivg.route('/stop/')
class Stop(Resource):
  def get (self):
    try:
        client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
        result_stop = client.predict(api_name="/end_process_with_update")
        while True:
            result_current = client.predict(api_name="/check_for_current_job")
            if result_current is None or len(result_current) == 0 or (len(result_current) > 0 and (result_current[0] is None or result_current[0] == "")):
                database.delete_wrong_entries(dbms)
                os.system("pkill -f uwsgi -9")
            else:
                time.sleep(1)
                result_stop = client.predict(api_name="/end_process_with_update")
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error', 500)

@limiter.limit("1/second")
@nsaivg.route('/generate/check/job/')
class GenerateCheck(Resource):
  def post (self):
    try:
        client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
        data = client.predict(api_name="/check_for_current_job")        
        text_ret = ""
        generation_id = database.select_config_by_skipped(dbms, 0)
        if generation_id is not None and data is not None and len(data) == 6:
            if data[4] != '':
                text_ret = text_ret + data[4] + "\n"
            if data[5] != '':
                splitted_data_5 = data[5].split("\n")
                for datah in splitted_data_5:
                    if "span" in datah:
                        to_add = datah.strip().replace("<span>","").replace("</span>","")
                        text_ret = text_ret + to_add + "&nbsp;"
            if text_ret != "":
                if "starting" not in text_ret.lower() and "clip vision" not in text_ret.lower() and len([path for path in Path(os.environ.get("OUTPUT_PATH")+"*").parent.glob('*.mp4')]) and len([path for path in Path(os.environ.get("OUTPUT_PATH")+"*").parent.glob('*.json')])> 0:
                    list_of_mp4 = glob.glob(os.environ.get("OUTPUT_PATH")+'*.mp4')
                    latest_mp4 = max(list_of_mp4, key=os.path.getctime)
                    latest_mp4_name = "_".join((Path(latest_mp4).stem).split("_")[:-1])
                    list_of_json = glob.glob(os.environ.get("OUTPUT_PATH")+'*.json')
                    latest_json = max(list_of_json, key=os.path.getctime)
                    latest_json_name = Path(latest_json).stem
                    if latest_mp4_name == latest_json_name:
                        response = send_file(latest_mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
                        response.headers['X-FramePack-File-Name'] = str(os.path.basename(latest_mp4)).encode('utf-8').decode('latin-1')
                    else: 
                        response = make_response('Job is starting', 202)
                        response.headers['X-FramePack-File-Name'] = str("").encode('utf-8').decode('latin-1')
                else:
                    response = make_response('Job is starting', 202)
                response.headers['X-FramePack-Check-Current-Job'] = text_ret.replace("\n","&nbsp;").encode('utf-8').decode('latin-1') 
                response.headers['X-FramePack-Generation-Id'] = str(generation_id[0]).encode('utf-8').decode('latin-1')
                return response
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
