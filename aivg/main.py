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

from dotenv import load_dotenv
from flask import Flask
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

def text2img(params: dict) -> dict:
    host = os.environ.get("FOOOCUS_ENDPOINT")
    result = requests.post(url=f"{host}/v1/generation/text-to-image",
                           data=json.dumps(params),
                           headers={"Content-Type": "application/json"})
    return result.json()

def generate_image(prompt):
    result = text2img({
        "prompt": prompt,
        "negative_prompt": os.environ.get("NEGATIVE_PROMPT"),
        "performance_selection": "Quality",
        "aspect_ratios_selection": "704*1344",
        "image_number": 1,
        "async_process": False})
    if len(result) > 0 and 'url' in result[0]:
        return result[0]['url']
    else:
        raise Exception("Result from Fooocus-API is None")

def download_png(url, file_path=os.environ.get("OUTPUT_PATH")):
    full_path = file_path + str(uuid.uuid4()) + ".png"
    urllib.request.urlretrieve(url, full_path)
    time.sleep(5)
    return full_path

def add_new_generation(video_len, mode, message=None, prompt=None):
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    result = client.predict(
		api_name="/check_for_current_job"
    )
    if len(result) > 0 and (result[0] is None or result[0] == ""):
        
        if prompt is None:
            message = random.choice(json.loads(os.environ.get('PROMPT_LIST'))) if message is None else message
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
                anything_llm_json = anything_llm_response.json()
                prompt = anything_llm_json["textResponse"].rstrip()
            else:
                raise Exception("Prompt from AnythingLLM is None")

        start_image = generate_image(prompt)
        photo_init = handle_file(download_png(start_image.replace("127.0.0.1", "172.17.0.1").replace("localhost", "172.17.0.1")))

        photo_end = None
        efi = 1
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
        gen_result = client.predict(
                selected_model="F1" if mode == 1 else "Original",
                param_1=photo_init,
                param_2=video,
                param_3=photo_end,
                param_4=round(random.uniform(0, 1.1), 1),
                param_5=prompt,
                param_6=os.environ.get("NEGATIVE_PROMPT"),
                param_7=efi,
                param_8=False,
                param_9=video_len,
                param_10=9, # window size
                param_11=50, # steps
                param_12=1,
                param_13=10,
                param_14=0.7,
                param_15="MagCache",
                param_16=25,
                param_17=0.15,
                param_18=0.25,
                param_19=5,
                param_20=0, #param_20=0.6,
                param_21=4,
                param_22="Noise",
                param_23=True,
                param_24=[],
                param_25=512,
                param_26=768,
                param_27=True,
                param_28=5,
                api_name="/handle_start_button"
        )
        if len(gen_result) > 0 and gen_result[1] is not None and gen_result[1] != "":
            job_id = gen_result[1]
            monitor_result = client.predict(
                    job_id=job_id,
                    api_name="/monitor_job"
            )
            if len(monitor_result) > 0 and 'video' in monitor_result[0]:
                mp4_name = os.path.basename(monitor_result[0]['video'])
                return mp4_name
            else:
                return None
        return prompt
    else:
        return False
    return None

def create_app():
    app = Flask(__name__)
    with app.app_context():
#        daemon = Thread(target=add_new_generation, args=(random.randint(5,60), 0,), daemon=True)
#        daemon.start()
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
  def post (self, mode = 0, message = None, video_len = 11):
    mp4_path = add_new_generation(video_len, mode, message=message)
    if mp4_path is None:
        return make_response('Error generating video', 500)
    elif mp4_path is False:
        return make_response('Another generation in progress', 206)
    return make_response(mp4_path, 200)

@limiter.limit("1/second")
@nsaivg.route('/generate/prompt/<string:prompt>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/')
@nsaivg.route('/generate/prompt/<string:prompt>/<int:mode>/<int:video_len>/')
class GeneratePrompt(Resource):
  def post (self, prompt = None, mode = 0, video_len = 11):
    mp4_path = add_new_generation(video_len, mode, prompt=prompt)
    if mp4_path is None:
        return make_response('Error generating video', 500)
    elif mp4_path is False:
        return make_response('Another generation in progress', 206)
    return make_response(mp4_path, 200)

@scheduler.task('interval', id='generate_loop', hours = 6)
def generate_loop():
    add_new_generation(random.randint(15,60), 0, message=None, prompt=None)

limiter.init_app(app)
scheduler.init_app(app)
scheduler.start()

if __name__ == '__main__':
  app.run()
