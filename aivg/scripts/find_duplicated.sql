select s.id, t.* 
from params s
join (
    select "has_input_image", "has_input_video", "model", "window_size", "steps", "cache_type", "tea_cache_steps", "tea_cache_rel_l1_thresh", "mag_cache_threshold", "mag_cache_max_consecutive_skips", "mag_cache_retention_ratio", "distilled_cfg_scale", "cfg_scale", "cfg_rescale", "lora", "lora_weight", "gen_photo", count(*) as qty
    from params
    group by "has_input_image", "has_input_video", "model", "window_size", "steps", "cache_type", "tea_cache_steps", "tea_cache_rel_l1_thresh", "mag_cache_threshold", "mag_cache_max_consecutive_skips", "mag_cache_retention_ratio", "distilled_cfg_scale", "cfg_scale", "cfg_rescale", "lora", "lora_weight", "gen_photo"
    having count(*) > 1
) t on s.has_input_image = t.has_input_image 
and s.has_input_video = t.has_input_video
and s.model = t.model
and s.window_size = t.window_size
and s.steps = t.steps
and s.cache_type = t.cache_type
and s.tea_cache_steps = t.tea_cache_steps
and s.tea_cache_rel_l1_thresh = t.tea_cache_rel_l1_thresh
and s.mag_cache_threshold = t.mag_cache_threshold
and s.mag_cache_max_consecutive_skips = t.mag_cache_max_consecutive_skips
and s.mag_cache_retention_ratio = t.mag_cache_retention_ratio
and s.distilled_cfg_scale = t.distilled_cfg_scale
and s.cfg_scale = t.cfg_scale
and s.cfg_rescale = t.cfg_rescale
and s.lora = t.lora
and s.lora_weight = t.lora_weight
and s.gen_photo = t.gen_photo
