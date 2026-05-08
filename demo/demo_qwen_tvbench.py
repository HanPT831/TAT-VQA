import os

os.chdir('../')

from utils.config import Config
import io

from models import VideoChat2_it_Qwen
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
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

from torchvision import transforms

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm
import webvtt
import re

import time
import decord
decord.bridge.set_bridge("torch")

def get_prompt(conv):
    ret = '<|im_start|>system\n' + conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + "\n" + message + "\n" + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = '<|im_start|>system\n' + conv.system + conv.sep
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
        torch.tensor([151645]).to("cuda:0")]
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


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

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

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

data_list = {
    "Action Count": ("action_count.json", "tvbench/action_count", "video", False),
    "Object Count": ("object_count.json", "tvbench/object_count", "video", False),
    "Action Sequence": ("action_sequence.json", "tvbench/action_sequence", "video", True),  # has start & end
    "Object Shuffle": ("object_shuffle.json", "tvbench/object_shuffle", "video", False),
    "Scene Transition": ("scene_transition.json", "tvbench/scene_transition", "video", False),
    "Action Localization": ("action_localization.json", "tvbench/action_localization", "video", True),  # has start & end
    "Action Antonym": ("action_antonym.json", "tvbench/action_antonym", "video", False),
    "Unexpected Action": ("unexpected_action.json", "tvbench/unexpected_action", "video", False),
    "Egocentric Sequence": ("egocentric_sequence.json", "tvbench/egocentric_sequence", "video", False),
    "Moving Direction": ("moving_direction.json", "tvbench/moving_direction", "video", False),
}

data_dir = "tvbench/json"

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }
    
#####################
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
save_path = ""
if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model = model.to(torch.device(cfg.device))
model = model.eval()

num_frame = 16
resolution = 224

dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)

def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        print_res=True,
        system_llm=False
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)

    chat = EasyDict({
            "system": system,
            "roles": ("<|im_start|>user", "<|im_start|>assistant\n"),
            "messages": [],
            "sep": "<|im_end|>\n"
        })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['answer']}")
    return llm_message

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

correct = 0
total = 0
res_list = []
acc_dict = {}

for example in tqdm(dataset):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0]
    acc_dict[task_type][1] += 1
    total += 1
    pred = infer_mvbench(
        example, 
        system="",
        question_prompt="\nOnly give the best option.",
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=True,
        system_llm=True
    )
    gt = example['answer']
    res_list.append({
        'pred': pred,
        'gt': gt
    })
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 30, task_type, '-' * 30)

with open(f"{save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

final_res = dict()
correct = 0
total = 0
for k, v in acc_dict.items():
    final_res[k] = v[0] / v[1] * 100
    correct += v[0]
    total += v[1]    
final_res['Avg'] = correct / total * 100

print(final_res)