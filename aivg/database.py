import os
import sys
import logging
from sqlalchemy.sql import func
from sqlalchemy import create_engine, insert, select, update, delete, Table, Column, Integer, String, Float, MetaData, DateTime, and_

SQLITE      = 'sqlite'
PARAMS     = 'params'

class Database:
  DB_ENGINE = {
      SQLITE: 'sqlite:////app/configs/{DB}'
  }

  # Main DB Connection Ref Obj
  db_engine = None
  def __init__(self, dbtype, username='', password='', dbname=''):
    dbtype = dbtype.lower()
    engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
    self.db_engine = create_engine(engine_url)

  metadata = MetaData()

  #skipped
  # 0 = empty
  # 1 = ok
  # 2 = ko

  #status
  # 0 = starting
  # 1 = sampling
  # 2 = upscaling
  # 3 = adding audio
  # 4 = completed

  params = Table(PARAMS, metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('skipped', Integer, nullable=False),
                Column('status', Integer, nullable=False),
                Column('requested_seconds', Integer, nullable=False),
                Column('has_input_image', Integer, nullable=False),
                Column('has_input_video', Integer, nullable=False),
                Column('model', String(50), nullable=False),
                Column('seed', Integer, nullable=False),
                Column('window_size', Integer, nullable=False),
                Column('steps', Integer, nullable=False),
                Column('cache_type', String(50), nullable=False),
                Column('tea_cache_steps', Integer, nullable=True),
                Column('tea_cache_rel_l1_thresh', Float, nullable=True),
                Column('mag_cache_threshold', Float, nullable=True),
                Column('mag_cache_max_consecutive_skips', Integer, nullable=True),
                Column('mag_cache_retention_ratio', Float, nullable=True),
                Column('distilled_cfg_scale', Float, nullable=False),
                Column('cfg_scale', Float, nullable=False),
                Column('cfg_rescale', Float, nullable=False),
                Column('lora', String, nullable=True),
                Column('lora_weight', Float, nullable=True),
                Column('gen_photo', Integer, nullable=False),
                Column('exec_time_seconds', Integer, nullable=True),
                Column('width', Integer, nullable=True),
                Column('height', Integer, nullable=True),
                Column('tms_insert', DateTime(timezone=True), server_default=func.now()),
                Column('tms_update', DateTime(timezone=True), onupdate=func.now())
                )

def create_db_tables(self):
  try:
    self.metadata.create_all(self.db_engine)
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)

def insert_wrong_config(self, config):
  generated_id = None
  try:
    loras_string = (', '.join(config['lora']) if config["lora"] is not None and len(config["lora"]) > 0 else None)
    stmt = insert(self.params).values(skipped=0).values(status=0).values(has_input_image=config['has_input_image']).values(has_input_video=config['has_input_video']).values(requested_seconds=config['requested_seconds']).values(model=config['model']).values(seed=config['seed']).values(window_size=config['window_size']).values(steps=config['steps']).values(cache_type=config['cache_type']).values(tea_cache_steps=config['tea_cache_steps']).values(tea_cache_rel_l1_thresh=config['tea_cache_rel_l1_thresh']).values(mag_cache_threshold=config['mag_cache_threshold']).values(mag_cache_max_consecutive_skips=config['mag_cache_max_consecutive_skips']).values(mag_cache_retention_ratio=config['mag_cache_retention_ratio']).values(distilled_cfg_scale=config['distilled_cfg_scale']).values(cfg_scale=config['cfg_scale']).values(cfg_rescale=config['cfg_rescale']).values(lora=loras_string).values(lora_weight=config['lora_weight']).values(gen_photo=config['gen_photo']).values(exec_time_seconds=config['exec_time_seconds']).values(width=config['width']).values(height=config['height']).prefix_with('OR IGNORE')
    compiled = stmt.compile()
    with self.db_engine.connect() as conn:
      result = conn.execute(stmt)
      conn.commit()
      generated_id = result.lastrowid
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)
  return generated_id

def select_config(self, config):
  try:
    value = None
    loras_string = (', '.join(config['lora']) if config["lora"] is not None and len(config["lora"]) > 0 else None)
    stmt = select(self.params.c.id,self.params.c.skipped,self.params.c.status).where(self.params.c.has_input_image==config['has_input_image'],self.params.c.has_input_video==config['has_input_video'],self.params.c.model==config['model'],self.params.c.window_size==config['window_size'],self.params.c.steps==config['steps'],self.params.c.cache_type==config['cache_type'],self.params.c.tea_cache_steps==config['tea_cache_steps'],self.params.c.tea_cache_rel_l1_thresh==config['tea_cache_rel_l1_thresh'],self.params.c.mag_cache_threshold==config['mag_cache_threshold'],self.params.c.mag_cache_max_consecutive_skips==config['mag_cache_max_consecutive_skips'],self.params.c.mag_cache_retention_ratio==config['mag_cache_retention_ratio'],self.params.c.distilled_cfg_scale==config['distilled_cfg_scale'],self.params.c.cfg_scale==config['cfg_scale'],self.params.c.cfg_rescale==config['cfg_rescale'],self.params.c.lora==loras_string,self.params.c.lora_weight==config['lora_weight'],self.params.c.gen_photo==config['gen_photo'],self.params.c.width==config['width'],self.params.c.height==config['height'])
    
    compiled = stmt.compile()
    with self.db_engine.connect() as conn:
      cursor = conn.execute(stmt)
      records = cursor.fetchall()

      if len(records) > 0:
        value = records[0]
      cursor.close()
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)
  finally:
    return value

def select_config_by_skipped(self, skipped):
  try:
    value = None
    stmt = select(self.params.c.id,self.params.c.skipped,self.params.c.status).where(self.params.c.skipped==skipped)
    
    compiled = stmt.compile()
    with self.db_engine.connect() as conn:
      cursor = conn.execute(stmt)
      records = cursor.fetchall()

      if len(records) > 0:
        value = records[0]
      cursor.close()
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)
  finally:
    return value

def update_config(self, config):
  try:
    value = []
    stmt = update(self.params).where(self.params.c.id==config["generation_id"])
    if "skipped" in config:
      stmt = stmt.values(skipped=config["skipped"])
    if "exec_time_seconds" in config:
      stmt = stmt.values(exec_time_seconds=config["exec_time_seconds"])
    if "status" in config:
      stmt = stmt.values(seed=config["seed"])
    if "status" in config:
      stmt = stmt.values(status=config["status"])
           
    compiled = stmt.compile()
    with self.db_engine.connect() as conn:
      result = conn.execute(stmt)
      conn.commit()
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)

def delete_wrong_entries(self):
  try:
    value = []
    stmt = delete(self.params).where(self.params.c.skipped==0)
           
    compiled = stmt.compile()
    with self.db_engine.connect() as conn:
      result = conn.execute(stmt)
      conn.commit()
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    raise(e)