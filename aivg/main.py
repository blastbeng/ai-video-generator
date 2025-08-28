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
        "image_number": 1,
        "image_seed": random.randint(0, 9223372036854775807),
        "async_process": False,
        "style_selections": [ "Fooocus V2", "Fooocus V2 (Optional)", "Fooocus Enhance", "Fooocus Sharp", "Fooocus Negative", "Fooocus Cinematic", "Cinematic Diva", "Fooocus Photograph" ]
        })
    if len(result) > 0 and 'url' in result[0]:
        return result[0]['url']
    else:
        raise Exception("Result from Fooocus-API is None")

def download_png(url, file_path=os.environ.get("OUTPUT_PATH")):
    full_path = file_path + str(uuid.uuid4()) + ".png"
    urllib.request.urlretrieve(url, full_path)
    time.sleep(5)
    return full_path

def add_audio_to_video(file, prompt, video_len):
    url = os.environ.get("MMAUDIO_ENDPOINT") + "/process"
    payload = {
        'prompt': prompt, 
        'negative_prompt': "", 
        'variant': "large_44k_v2", 
        'duration': str(video_len)
    }
    with  open(file,'rb') as file:
        response = requests.post(url, data=payload, files={'video': file})
        if response.status_code == 200:
            return response.content
    return None

def add_new_generation(video_len, mode=1, message=None, prompt=None):
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
                
                prompt = anything_llm_json = anything_llm_response.json()["textResponse"].rstrip()
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
                raise Exception("Error getting response from AnythingLLM")
        else:
            prompt_image = prompt
        
        start_image = generate_image(prompt_image)
        photo_init = handle_file(download_png(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1")))

        photo_end = None
        efi = 1
        seed = random.randint(0, 9223372036854775807)
        ##if mode == 0:
        ##    end_image = generate_image(message)
        ##    photo_end = handle_file(download_png(end_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1")))
        ##    efi = random.randint(0,9223372036854775807)

        #photo_init = handle_file(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"))
        #photo_end = handle_file(end_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1"))
        video = None
        #if is_video:
        #    content = await update.message.effective_attachment.get_file()
        #    video = {"video":handle_file(content.file_path)}
        #elif not from_cmd:
        #    content = await update.message.effective_attachment[-1].get_file()
        #    photo = handle_file(content.file_path)
        ##original = "Original" if photo_end is None else "Original with Endframe"
        loras = ['hunyuan_video_accvid_5_steps_lora_rank16_fp8_e4m3fn', 'hyvid_I2V_lora_hair_growth', 'hyvideo_FastVideo_LoRA-fp8', 'hunyuan_video_720_cfgdistill_fp8_e4m3fn', 'hyvid_I2V_lora_embrace', 'hunyuan_video_FastVideo_720_fp8_e4m3fn']
        gen_result = client.predict(
                selected_model="F1" if mode == 1 else "Original",
                param_1=photo_init,
                param_2=video,
                param_3=photo_end,
                param_4=efi,
                param_5=prompt,
                param_6=os.environ.get("NEGATIVE_PROMPT"),
                param_7=seed,
                param_8=False,
                param_9=video_len,
                param_10=9, # window size
                param_11=40, # steps
                param_12=1,
                param_13=10,
                param_14=0, #param_14=0.7,
                param_15="MagCache",
                param_16=25,
                param_17=0.15,
                param_18=0.25,
                param_19=5,
                param_20=0.2, #param_20=0.6,
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
                file = os.environ.get("OUTPUT_PATH") + os.path.basename(monitor_result[0]['video'])
                result_upscale = client.predict(
                        video_path={"video":handle_file(file)},
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
                    mp4 = add_audio_to_video(file_upscaled, prompt, video_len)
                    return mp4
                else:
                    return None
            else:
                return None
        return prompt
    else:
        return False
    return None

def create_app():
    app = Flask(__name__)
    with app.app_context():
        #daemon = Thread(target=add_new_generation, args=(random.randint(5,60),), daemon=True)
        #daemon.start()
        remove_directory_tree(Path(os.environ.get("OUTPUT_PATH")))
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
@nsaivg.route('/generate/enhance/<int:mode>/<int:video_len>/')
@nsaivg.route('/generate/enhance/<int:mode>/<int:video_len>/<string:message>/')
class GenerateMessage(Resource):
  def post (self, mode = 1, message = None, video_len = 11):
    try:
        mp4 = add_new_generation(video_len, mode=mode, message=message)
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        return send_file(BytesIO(mp4), attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
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
  def post (self, prompt = None, mode = 1, video_len = 11):
    try:
        mp4 = add_new_generation(video_len, mode=mode, prompt=prompt)
        if mp4 is None:
            return make_response('Error generating video', 500)
        elif mp4 is False:
            return make_response('Another generation in progress', 206)
        return send_file(BytesIO(mp4), attachment_filename=str(uuid.uuid4()) + '.mp4', mimetype='video/mp4')
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      return make_response('Error generating video', 500)

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
        return make_response('Stopping current generation...', 206)
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
