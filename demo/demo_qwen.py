import os
from utils.config import Config
import io
from pathlib import Path
import csv
from models import VideoChat2_it_Qwen
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import time
import decord
decord.bridge.set_bridge("torch")
os.chdir('../')

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + "\n" + message + "\n" + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + "\n" + message
        else:
            if message:
                ret += role + "\n" + message + "\n" + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.qwen_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.qwen_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([151643]).to("cuda:0"),
        torch.tensor([151645]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.qwen_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    output_text = model.qwen_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('<|im_end|>')[0]
    conv.messages[-1][1] = output_text + '<|im_end|>'
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, aot="forward"):
    """
    Args:
        video_path (str): path to video
        num_segments (int): number of sampled frames
        return_msg (bool): whether to return textual sampling info
        resolution (int): input resolution
        aot (str): "forward", "backward", or "random"
    
    Returns:
        torch_imgs: shape depends on your transform pipeline
        msg (optional): sampling message
        aot_label (optional): 1 for forward, 0 for backward
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # decide direction
    if aot == "random":
        aot = random.choice(["forward", "backward"])
    elif aot not in ["forward", "backward"]:
        raise ValueError(f"Invalid aot={aot}, expected 'forward', 'backward', or 'random'.")

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    # load sampled frames
    images_group = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)

    # reverse frame order for Arrow-of-Time backward samples
    if aot == "backward":
        images_group = images_group[::-1]

    torch_imgs = transform(images_group)

    # label: 1 = forward, 0 = backward
    aot_label = 1 if aot == "forward" else 0

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = [round(f / fps, 1) for f in frame_indices]

        if aot == "backward":
            sec = sec[::-1]

        sec_str = ", ".join([str(x) for x in sec])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec_str} seconds. Direction: {aot}."
        return torch_imgs, msg, aot_label
    else:
        return torch_imgs, aot_label

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

config_file = "configs/config_qwen.json"
cfg = Config.from_file(config_file)

cfg.model.vision_encoder.num_frames = 16
model = VideoChat2_it_Qwen(config=cfg.model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)
model.qwen_model = get_peft_model(model.qwen_model, peft_config)

state_dict = torch.load("", "cpu", weights_only=False)

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False,)
else:
    msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model = model.to(torch.device(cfg.device))
model = model.eval()

num_frame = 16
resolution = 224
files = []
acc = 0
results = []
output_json = ""
with open("aot_ppb/test.csv", mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        files.append({'path':row[0], 'label':row[1]})
for idx, f in enumerate(tqdm(files)):
    id2label = ['backward', 'forward']
    gt_label = id2label[int(f['label'])]

    vid, msg, _ = load_video(f['path'], 
                             num_segments=num_frame, 
                             return_msg=True, 
                             resolution=resolution, 
                             aot=gt_label)
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")

    img_list = []
    with torch.no_grad():
        image_emb, _ = model.encode_img(video, "")

    img_list.append(image_emb)

    chat = EasyDict({
        "system": "",
        "roles": ("<|im_start|>user", "<|im_end|>\n<|im_start|>assistant\n"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> <|im_end|>\n"])

    ask("Question: Based on the temporal cues, is the video forward or backward?", chat)
    llm_message = answer(conv=chat, model=model, 
                         do_sample=False, 
                         img_list=img_list, 
                         max_new_tokens=16, 
                         print_res=True, 
                         answer_prompt="Answer:")[0]
    
    raw_prediction = llm_message.strip()
    pred_lower = raw_prediction.lower()

    if pred_lower.startswith("forward"):
        pred_label = "forward"
    elif pred_lower.startswith("backward"):
        pred_label = "backward"
    else:
        pred_label = "unknown"

    correct = pred_label == gt_label
    if correct:
        acc += 1

    record = {
        "index": idx,
        "video_path": f["path"],
        "ground_truth": gt_label,
        "prediction": pred_label,
        "raw_response": raw_prediction,
        "correct": correct,
        "sampling_message": msg
    }

    results.append(record)
    save_data = {
        "summary": {
            "num_samples": len(results),
            "num_correct": acc,
            "accuracy": acc / len(results)
        },
        "results": results
    }

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(save_data, jf, ensure_ascii=False, indent=2)

    print(f"prediction: {raw_prediction}")
    print(f"running acc: {acc}/{len(results)} = {acc / len(results):.4f}")

final_acc = acc / len(files)

print(f"Final accuracy: {final_acc:.4f}")
print(f"Saved results to: {output_json}")