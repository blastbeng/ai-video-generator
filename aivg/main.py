import logging
import os
import random
import shutil
import json
import sys
import uuid
import asyncio
import aiohttp

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

def create_app():
    app = Flask(__name__)
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
@nsaivg.route('/generate/enhance/<int:video_len>/')
@nsaivg.route('/generate/enhance/<int:video_len>/<string:message>/')
class GenerateMessage(Resource):
  def post (self, message = None, video_len = 5):
    daemon = Thread(target=asyncio.run, args=(add_new_generation(video_len, message=message),), daemon=True, name="add_new_generation_"+str(uuid.uuid4()))
    daemon.start()
    return make_response('Adding a new generation to the queue', 200)

@limiter.limit("1/second")
@nsaivg.route('/generate/prompt/<int:video_len>/')
@nsaivg.route('/generate/prompt/<int:video_len>/<string:prompt>/')
class GeneratePrompt(Resource):
  def post (self, prompt = None, video_len = 5):
    daemon = Thread(target=asyncio.run, args=(add_new_generation(video_len, prompt=prompt),), daemon=True, name="add_new_generation_"+str(uuid.uuid4()))
    daemon.start()
    return make_response('Adding a new generation to the queue', 200)
    
async def add_new_generation(video_len, message=None, prompt=None):
    if prompt is None:
        message = random.choice(json.loads(os.environ.get('PROMPT_LIST'))) if message is None else message
        data = {
            "message": message,
            "mode": "chat"
        }
        headers = {
            'Authorization': 'Bearer ' + os.environ.get("ANYTHING_LLM_API_KEY")
        }
        connector = aiohttp.TCPConnector(force_close=True)
        anything_llm_url = os.environ.get("ANYTHING_LLM_ENDPOINT") + "/api/v1/workspace/" + os.environ.get("ANYTHING_LLM_WORKSPACE") + "/chat"
        async with aiohttp.ClientSession(connector=connector) as anything_llm_session:
            async with anything_llm_session.post(anything_llm_url, headers=headers, json=data) as anything_llm_response:
                if (anything_llm_response.status == 200):
                    anything_llm_json = await anything_llm_response.json()
                    prompt = anything_llm_json["textResponse"].rstrip()
                else:
                    raise Exception("Prompt from AnythingLLM is None")
            await anything_llm_session.close()
    photo = None
    video = None
    #if is_video:
    #    content = await update.message.effective_attachment.get_file()
    #    video = {"video":handle_file(content.file_path)}
    #elif not from_cmd:
    #    content = await update.message.effective_attachment[-1].get_file()
    #    photo = handle_file(content.file_path)
    client = Client(os.environ.get("FRAMEPACK_ENDPOINT"))
    result = client.predict(
            selected_model="F1",
            param_1=photo,
            param_2=video,
            param_3=None,
            param_4=1,
            param_5=prompt,
            param_6="",
            param_7=random.randint(0,9223372036854775807),
            param_8=False,
            param_9=video_len,
            param_10=18,
            param_11=50,
            param_12=1,
            param_13=10,
            param_14=0.7,
            param_15="MagCache",
            param_16=25,
            param_17=0.15,
            param_18=0.25,
            param_19=5,
            param_20=0.6,
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
    #monitor_result = client.predict(
    #    job_id=result[1],
    #    api_name="/monitor_job"
    #)

#@scheduler.task('interval', id='test', seconds = 5)
#def test():
#  print("test")

limiter.init_app(app)
scheduler.init_app(app)
scheduler.start()

if __name__ == '__main__':
  app.run()
