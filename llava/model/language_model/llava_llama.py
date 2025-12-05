#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    ...

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass  # ✅ 新增

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# ✅ 新增自定义输出类
@dataclass
class LlavaCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    扩展的输出类，添加 router_diversity_loss
    """
    router_diversity_loss: Optional[torch.FloatTensor] = None


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def _compute_router_diversity_loss(self):
        """
        计算 router diversity loss
        从 llava_arch.encode_images 中收集的 diversity losses
        """
        if not self.training:
            return None
            
        # 检查是否启用了 router
        use_router = getattr(self.model.config, 'use_router', False)
        if not use_router:
            return None
        
        # 从 model 中获取收集的 diversity losses
        if hasattr(self.model, '_router_diversity_losses'):
            losses = self.model._router_diversity_losses
            if losses:
                avg_loss = torch.stack(losses).mean()
                # print(f"  🔧 Computed diversity loss: {avg_loss.item():.6f} from {len(losses)} samples")
                # 清空列表，为下一个 batch 准备
                self.model._router_diversity_losses = []
                return avg_loss
        
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:

        # print(f"\n{'='*60}")
        # print(f"🔍 llava_llama.forward 开始:")
        # print(f"  self.training: {self.training}")
        # print(f"{'='*60}\n")

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # ✅ 调用父类的 forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        task_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # 3️⃣ 计算 router diversity loss
        router_diversity_loss = None
        total_loss = task_loss

        if self.training and task_loss is not None and labels is not None:
            router_diversity_loss = self._compute_router_diversity_loss()
            
            if router_diversity_loss is not None:
                # ✅ 从config中读取diversity_weight
                diversity_weight = getattr(self.config, 'diversity_weight', 1.0)
                
                # ✅ 计算加权的diversity loss
                weighted_diversity_loss = diversity_weight * router_diversity_loss
                
                # ✅ 累加到总损失
                total_loss = task_loss + weighted_diversity_loss
                
                # print(f"\n{'='*60}")
                # print(f"  Task Loss:            {task_loss.item():.6f}")
                # print(f"  Diversity Loss (raw): {router_diversity_loss.item():.6f}")
                # print(f"  Diversity Weight:     {diversity_weight:.2f}")
                # print(f"  Diversity Loss (weighted): {weighted_diversity_loss.item():.6f}")
                # print(f"  Total Loss:           {total_loss.item():.6f}")
                # print(f"{'='*60}\n")
                
        # 5️⃣ 返回结果（带上 diversity loss）
        if isinstance(outputs, dict):
            outputs['loss'] = total_loss
            outputs['router_diversity_loss'] = router_diversity_loss
            return outputs
        else:
            return LlavaCausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_diversity_loss=router_diversity_loss,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)