from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
from .models_seggpt import seggpt_vit_large_patch16_input896x448
from .seggpt_engine import inference_image_pil
import torch
import numpy as np
import math
import comfy.utils
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

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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
        device = comfy.model_management.get_torch_device()
        model = prepare_model().to(device)
        prompt = Image.fromarray(np.clip(255. * prompt[0].cpu().numpy(), 0, 255).astype(np.uint8))
        promptMask = Image.fromarray(np.clip(255. * promptMask[0].cpu().numpy(), 0, 255).astype(np.uint8))
        results = []
        resultsPrev = []
        ii = 0
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            rImg,rImgPrev = inference_image_pil(model,device,img,[prompt],[promptMask])
            rNPImg = pil2tensor(rImg)
            rNPImgPrev = pil2tensor(rImgPrev)
            results.append(rNPImg)
            resultsPrev.append(rNPImgPrev)
            ii = ii + 1
            print(f"segGPT ok:{ii}")
        r1 = torch.cat(results, dim=0)
        r2 = torch.cat(resultsPrev, dim=0)
        del model
        return (r1,r2)
        
'''      
seggpt_test = Image.open("seggpt_test.jpg")
prompt = Image.open("prompt.jpg")
promptMask = Image.open("promptMask.jpg")
device = torch.device("cuda")
model = prepare_model().to(device)
mask,preview = inference_image_pil(model,device,seggpt_test,[prompt],[promptMask])
mask.save('mask.jpg')
preview.save('preview.jpg')

#inference_image(model, device, "seggpt_test.jpg", ["prompt.jpg"], ["promptMask.jpg"], 'mask.jpg','preview.jpg')
'''

