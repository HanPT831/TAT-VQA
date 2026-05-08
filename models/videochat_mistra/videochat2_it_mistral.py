import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

from ..blip2.blip2 import Blip2Base, disabled_train
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import math
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VideoChat2_it_mistral(Blip2Base):
    """
    VideoChat2 model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        qformer_model_path = config.get("qformer_model_path", None)
        mistral_model_path = config.get("mistral_model_path")
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
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        # debug
        self.debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", True)
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

        logger.info('Loading Mistral')
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
        self.mistral_tokenizer.padding_side = "left"
        if not self.mistral_tokenizer.pad_token:
            logger.info("Set pad_token")
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

        if self.debug:
            logger.info("Debug mode, build small Mistral")
            mistral_config = AutoConfig.from_pretrained(mistral_model_path)
            mistral_config.hidden_size = 512
            mistral_config.intermediate_size = 2048
            mistral_config.num_attention_heads = 8
            mistral_config.num_hidden_layers = 12
            mistral_config.torch_dtype = torch.float16
            self.mistral_model = AutoModelForCausalLM.from_config(mistral_config)
        else:
            if use_flash_attention:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                    # use_flash_attention_2=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                )

        logger.info("freeze Mistral")
        for _, param in self.mistral_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading Mistral Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj", "lm_head"]
            )
            self.mistral_model = get_peft_model(self.mistral_model, peft_config)
            self.mistral_model.print_trainable_parameters()
        lm_dim = self.mistral_model.config.hidden_size
        self.mistral_proj = nn.Sequential(
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
            image_embeds = self.mistral_proj(image_embeds)   # [B, T, L_new, C]
            inputs_mistral = image_embeds.reshape(B, T_new * L_new, -1)   # [B, T*L_new, C]
        return inputs_mistral, use_image
        
    def _get_text_len(self, text):
        return self.mistral_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

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
            p_before_tokens = self.mistral_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.mistral_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.human_start + " "
            sep2 = " " + self.human_end + " "
            raw_text = p_after.split(sep2)
            for idx in range(0, len(raw_text) - 1):
                raw_text[idx] = raw_text[idx] + sep2
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # [target] "xxxxx. </s>"
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())

            if self.debug:  # Inspect and check the correctness of masking
                z = answer_targets[0].clone()
                z = torch.where(z == -100, self.mistral_tokenizer.unk_token_id, z)
                logger.info(self.mistral_tokenizer.decode(z))
                
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.mistral_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.mistral_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.mistral_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.mistral_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.mistral_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False, # current flash_attn2 dows not support padding=right for mistral
            )
    
        return dict(
            loss=outputs.loss,
        )
