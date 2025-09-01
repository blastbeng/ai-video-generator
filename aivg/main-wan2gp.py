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

def add_audio_to_video(file_path, prompt, video_len):
    url = os.environ.get("MMAUDIO_ENDPOINT") + "/process"
    payload = {
        'prompt': prompt, 
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

def parse_prompt(message, prompt):
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

    return prompt, prompt_image

def add_new_generation_framepack(video_len, mode, gen_photo, message, prompt, image, video):
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    result = client.predict(
		api_name="/check_for_current_job"
    )
    if result is not None and len(result) > 0 and (result[0] is None or result[0] == ""):
        
        prompt, prompt_image = parse_prompt(message, prompt)
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
        efi = 1
        loras = [
            'hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn', 
            'HunyuanVideo_dashtoon_keyframe_lora_converted_comfy_bf16', 
            'hyvid_I2V_lora_hair_growth', 
            'hyvideo_FastVideo_LoRA-fp8', 
            'hyvid_I2V_lora_embrace', 
            'HunyuanVideo_dashtoon_keyframe_lora_converted_bf16'
        ]
        seed = random.randint(0, 9223372036854775807)

        generated_video = get_video(client, mode, photo_init, video_init, efi, prompt, seed, loras, 0, video_len)
        return generated_video
    else:
        return False
    return None

def get_video(client, mode, photo_init, video_init, efi, prompt, seed, loras, actual_seconds, requested_seconds):
    model = "F1" if mode else "Original"
    if video_init is not None:
        model = "Video F1" if mode else "Video"
    gen_result = client.predict(
                selected_model=model,
                param_1=handle_file(photo_init) if photo_init is not None else None,
                param_2=({"video":handle_file(video_init)}) if video_init is not None else None,
                param_3=None,
                param_4=efi,
                param_5=prompt,
                param_6=os.environ.get("NEGATIVE_PROMPT"),
                param_7=seed,
                param_8=False,
                param_9=seconds_to_gen,
                param_10=9, # window size
                param_11=25, # steps
                param_12=1,
                param_13=10,
                param_14=0, #param_14=0.7,
                param_15="None",
                param_16=30,
                param_17=0.15,
                param_18=0.35,
                param_19=5,
                param_20=0, #param_20=0.6,
                param_21=4,
                param_22="Noise",
                param_23=True,
                param_24=loras, #param_24=random.sample(loras, random.randint(1, len(loras))),
                param_25=512, #param_25=512,
                param_26=768, #param_26=768,
                param_27=True,
                param_28=5,
                param_30=1, 
                param_31=1, 
                param_32=1, 
                param_33=1, 
                param_34=1, 
                param_35=1, 
                api_name="/handle_start_button"
        )
    if gen_result is not None and len(gen_result) > 0 and gen_result[1] is not None and gen_result[1] != "":
        job_id = gen_result[1]
        monitor_result = client.predict(
                job_id=job_id,
                api_name="/monitor_job"
        )
        if monitor_result is not None and len(monitor_result) > 0 and 'video' in monitor_result[0]:
            generated_video = (os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video']))
            if generated_video is not None:
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
                    file_upscaled = os.environ.get("OUTPUT_PATH") + "postprocessed_output/saved_videos/" + os.path.basename(result_upscale[0]['video'])
                    mp4 = add_audio_to_video(file_upscaled, prompt_image, video_len)
                    return mp4
        return None
    return None

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
        #remove_directory_tree(Path(os.environ.get("OUTPUT_PATH")))
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
        future = executor.submit(add_new_generation_framepack, video_len, (True if mode == 1 else False), (True if gen_photo == 1 else False), message, None, request.files["image"].read() if "image" in request.files else None, request.files["video"].read() if "video" in request.files else None)
        mp4 = future.result()
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        return send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
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
        future = executor.submit(add_new_generation_framepack, video_len, (True if mode == 1 else False), (True if gen_photo == 1 else False), None, prompt, request.files["image"].read() if "image" in request.files else None, request.files["video"].read() if "video" in request.files else None)
        mp4 = future.result()
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        return send_file(mp4, attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
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
