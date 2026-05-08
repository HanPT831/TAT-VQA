import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

from ..blip2.blip2 import Blip2Base, disabled_train
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

import math
import torch.nn.functional as F


class VideoChat2_it_Qwen(Blip2Base):
    """
    VideoChat2 model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        qwen_model_path = config.get("qwen_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")  
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # MLP
        self.spatial_pool_shape = config.get("spatial_pool_shape", None)
        self.num_temporal_pool_tokens = config.get("num_temporal_pool_tokens", None)
        self.temporal_pool_mode = config.get("temporal_pool_mode", "avg")
        vis_dim = config.vision_encoder.encoder_embed_dim
        dropout = config.get("temporal_dropout", 0.2)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.return_index = config.get("return_index", -2)  
        self.begin_signal = "<|im_start|>"
        self.role = ("user", "assistant")
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        # debug
        debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.tokenizer.padding_side = "left"
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm = self.init_vision_encoder_internvideo(config)

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        logger.info('Loading qwen')
        # problem: do we need to set truncation_side="left"?
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
        # self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        self.qwen_tokenizer.padding_side = "left"

        if use_flash_attention:
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                # use_flash_attention_2=True,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        else:
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        logger.info("freeze qwen")
        for _, param in self.qwen_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading qwen Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj", "lm_head"]
            )
            self.qwen_model = get_peft_model(self.qwen_model, peft_config)
            self.qwen_model.print_trainable_parameters()

        lm_dim = self.qwen_model.config.hidden_size
        self.qwen_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, lm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lm_dim, lm_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(lm_dim, lm_dim),
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def apply_adaptive_avg_pooling(self, x, shape=(12, 12)):
        """
        x: [B, N, C], where N = H * W
        returns: [B, shape[0] * shape[1], C]
        """
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens, f"num_tokens={num_tokens} is not a square number"

        x = x.permute(0, 2, 1).reshape(b, c, h, h)   # [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, shape)          # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)             # [B, H'*W', C]
        return x

    def spatial_pool_video_tokens(self, x, shape=(4, 4)):
        """
        x: [B, T, L, C], where L = H * W
        returns: [B, T, L_new, C]
        """
        B, T, L, C = x.shape
        x = x.reshape(B * T, L, C)
        x = self.apply_adaptive_avg_pooling(x, shape)      # [B*T, L_new, C]
        L_new = x.shape[1]
        x = x.reshape(B, T, L_new, C)
        return x

    def temporal_pool_video_tokens(self, x, num_pool_tokens=8, mode="avg"):
        """
        x: [B, T, L, C]
        returns: [B, T_new, L, C]
        pools only along the temporal dimension
        """
        B, T, L, C = x.shape
        T_new = min(num_pool_tokens, T)

        if T_new == T:
            return x

        # [B, T, L, C] -> [B*L, C, T]
        x = x.permute(0, 2, 3, 1).reshape(B * L, C, T)

        if mode == "avg":
            x = F.adaptive_avg_pool1d(x, T_new)          # [B*L, C, T_new]
        elif mode == "max":
            x = F.adaptive_max_pool1d(x, T_new)          # [B*L, C, T_new]
        else:
            raise ValueError(f"Unsupported temporal pooling mode: {mode}")

        # [B*L, C, T_new] -> [B, T_new, L, C]
        x = x.reshape(B, L, C, T_new).permute(0, 3, 1, 2).contiguous()
        return x

    def encode_img(self, image, instruction, return_index=-2):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.vision_encoder(image, use_image, x_vis_return_idx=return_index)
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
            image_embeds = image_embeds.reshape(B, T, L, C)
            
            if self.spatial_pool_shape is not None:
                image_embeds = self.spatial_pool_video_tokens(
                    image_embeds,
                    shape=self.spatial_pool_shape
                )

            if self.num_temporal_pool_tokens is not None:
                image_embeds = self.temporal_pool_video_tokens(
                    image_embeds,
                    num_pool_tokens=self.num_temporal_pool_tokens,
                    mode=self.temporal_pool_mode,
                )

            B, T_new, L_new, C = image_embeds.shape
            image_embeds = self.qwen_proj(image_embeds)   # [B, T, L_new, C]
            inputs_qwen = image_embeds.reshape(B, T_new * L_new, -1)   # [B, T*L_new, C]


        return inputs_qwen, use_image
        
    def _get_text_len(self, text):
        return self.qwen_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input, instruction):
        img_embeds, use_image = self.encode_img(image, instruction, return_index=self.return_index)
        batch_size, img_len, _ = img_embeds.shape

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.qwen_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.qwen_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.qwen_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.qwen_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.qwen_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.qwen_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + "\n"
            sep2 = self.begin_signal + self.role[1] + "\n"
            raw_text = p_after.split(sep2)
            for idx in range(1, len(raw_text)):
                raw_text[idx] = sep2 + raw_text[idx]

            answer_targets = p_after_tokens.input_ids.clone()
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system)
            sep_len = self._get_text_len(sep1)
            cur_len = self._get_text_len(raw_text[0])

            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text)
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1))
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1])
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.qwen_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.qwen_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.qwen_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        # inputs_embeds[:, :1] = self.qwen_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, :(input_len)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len):(input_len)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,
            )
    
        return dict(
            loss=outputs.loss,
        )
