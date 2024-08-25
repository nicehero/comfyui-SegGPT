import os
import torch
import gradio as gr
import sys
import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
import models_seggpt

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

#command = f'python \
#  util/painter_inference_demo2.py'
#os.system(command)

def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model
'semantic'
device = torch.device('cuda')
model = prepare_model('seggpt_vit_large.pth', 'seggpt_vit_large_patch16_input896x448','instance' ).to(device)
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Cache(list):
    def __init__(self, max_size=0):
        super().__init__()
        self.max_size = max_size

    def append(self, x):
        if self.max_size <= 0:
            return
        super().append(x)
        if len(self) > self.max_size:
            self.pop(0)


@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def inference_image(model, device, img_path, img2_paths, tgt2_paths):
    res, hres = 448, 448

    image = img_path#Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        img2 = img2_path#Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.
        tgt2 = tgt2_path#Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()
    output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))
    return output

def predict(imagemask,image2):
    img2 = imagemask["image"].convert("RGB")
    tgt2 = imagemask["mask"].convert("RGB")
    img = image2
    return inference_image(model, device, img, [img2], [tgt2])
    
with gr.Blocks(css='.fixed-height.svelte-rlgzoo {height: 100%;}') as demo:
    with gr.Column():
        with gr.Row():
            inp = [gr.ImageMask(type="pil").style(height=400),gr.Image(type="pil").style(height=400)]
            out = gr.Image(type="pil")
        btn = gr.Button("Run")
    btn.click(fn=predict, inputs=inp, outputs=out)
    
demo.launch(server_name="192.168.0.100",server_port=27871)
