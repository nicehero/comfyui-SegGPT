import os
f = r'C:\telegramDownload\oldWork\定制\jc\jc_02'
if not os.path.exists(f'{f}/preview'):
  os.makedirs(f'{f}/preview')
if not os.path.exists(f'{f}/preview_'):
  os.makedirs(f'{f}/preview_')

command = f'python seggpt_inference_batch.py \
--input_image_path {f}/frames \
--prompt_image {f}/prompt.jpg {f}/prompt2.jpg {f}/prompt3.jpg \
--prompt_target {f}/promptMask.jpg {f}/promptMask2.jpg {f}/promptMask3.jpg \
--output_dir {f}/preview'

command = f'python seggpt_inference_batch.py \
--input_image_path {f}/frames2 \
--prompt_image {f}/prompt.jpg \
--prompt_target {f}/promptMask.jpg \
--output_dir {f}/preview'
os.system(command)
