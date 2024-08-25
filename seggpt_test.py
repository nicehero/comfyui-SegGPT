from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
from models_seggpt import seggpt_vit_large_patch16_input896x448
from seggpt_engine import inference_image_pil,inference_image
import torch
import numpy as np
import math
import sys

INT = ("INT", {"default": 512,
               "min": -10240,
               "max": 10240,
               "step": 64})
def get_image_size(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    # size = size.movedim(1, -1)
    return size

def convert_to_nearest_multiple_of_64(num):
    return ((num + 31) // 64) * 64

import os

# 获取当前文件的目录

def prepare_model(seg_type='semantic'):
    # build model
    model = seggpt_vit_large_patch16_input896x448()
    model.seg_type = seg_type
    # load model
    current_directory = os.path.dirname(os.path.abspath(__file__))
    checkpoint = torch.load(os.path.join(current_directory,'seggpt_vit_large.pth'), map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

class SegGPT:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("IMAGE",),
                "promptMask": ("IMAGE",),
                }}

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("MASKS", "PREVIEW",)
    FUNCTION = "doSegGPT"

    CATEGORY = "SegGPT"

    def doSegGPT(self, images, prompt,promptMask):
        model = prepare_model().to(device)
        prompt = Image.fromarray(np.clip(255. * prompt[0].cpu().numpy(), 0, 255).astype(np.uint8))
        promptMask = Image.fromarray(np.clip(255. * promptMask[0].cpu().numpy(), 0, 255).astype(np.uint8))
        results = []
        resultsPrev = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            rNPImg,rNPImgPrev = np.array(inference_image_pil(model,comfy.model_management.get_torch_device(),img,[prompt],[promptMask])) / 255.
            results.append(rNPImg)
            resultsPrev.append(rNPImgPrev)
        r1 = np.array(results)
        r1 = np.array(resultsPrev)
        return (r1,r2)
      
