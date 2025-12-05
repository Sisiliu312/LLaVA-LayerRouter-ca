#    Copyright 2023 Haotian Liu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_cross_attn, build_layer_router

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            # print("==========================Building vision tower================")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if getattr(config, 'use_ca', False):
                self.ca = build_cross_attn(config)
            
            if getattr(config, 'use_router', False):
                self.layer_router = build_layer_router(config)


            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_cross_attn(self):
        return self.ca
    
    def get_layer_router(self):
        return self.layer_router

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        self.config.use_ca = getattr(model_args, 'use_ca', False)
        self.config.use_router = getattr(model_args, 'use_router', False)

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if self.config.use_ca:
            if getattr(self, 'ca', None) is None:
                self.ca = build_cross_attn(self.config)

        if self.config.use_router:
            if getattr(self, 'layer_router', None) is None:
                self.layer_router = build_layer_router(self.config)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            if self.config.use_ca and hasattr(self, 'ca') and self.ca is not None:
                ca_weights = get_w(mm_projector_weights, 'ca')
                if ca_weights:
                    self.ca.load_state_dict(ca_weights)

            if self.config.use_router and hasattr(self, 'layer_router') and self.layer_router is not None:
                router_weights = get_w(mm_projector_weights, 'layer_router')
                if router_weights:
                    self.layer_router.load_state_dict(router_weights)
        
        



def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_cross_attn(self):
        return self.get_model().get_cross_attn()

    def get_layer_router(self):
        return self.get_model().get_layer_router()


    # def encode_images(self, images, text_token):
    #     # 获取24层特征
    #     image_features, image_forward_outs = self.get_model().get_vision_tower()(images)

    #     # 处理text
    #     combined_text = torch.cat(text_token, dim=0)
    #     if combined_text.dim() == 2:
    #         combined_text = combined_text.unsqueeze(0)

    #     # 初始化最终输出
    #     batch_size, text_len, dim = combined_text.shape
    #     batch_size = image_features.shape[0]
    #     combined_features = torch.zeros(batch_size, text_len, dim, device=combined_text.device, dtype=combined_text.dtype)
    #     # print("image_features shape:",image_features.shape)
    #     # image_features = dense_connector(image_features, image_forward_outs, text_token)
    #     for i in [3,8,13,18,23]:
    #     # 获取当前层特征 [batch, num_patches, dim]
    #         # layer_feat = image_forward_outs.hidden_states[i]
    #         # layer_feat = image_forward_outs.hidden_states[i].half()
    #         layer_feat = image_forward_outs.hidden_states[i][:, 1:].to(image_features.dtype)
    #         # layer_feat = image_forward_outs.hidden_states[i].to(self.get_model().mm_projector.weight.dtype)
    #         # print("layer_feat type:", type(layer_feat))
    #         layer_features = self.get_model().mm_projector(layer_feat)
    #         attended = self.get_model().ca(combined_text, layer_features)
    #         combined_features += attended
    #         # print("layer_feat type:", type(layer_feat))
    #     return combined_features

    # def encode_images(self, images, text_token):
    #     """
    #     Batched version - assumes all samples select the same top-5 layers.
    #     More efficient but less flexible.
    #     """
    #     # 获取24层特征
    #     image_features, image_forward_outs = self.get_model().get_vision_tower()(images)

    #     use_router = getattr(self.get_model().config, 'use_router', False)
    #     use_ca = getattr(self.get_model().config, 'use_ca', False)

    #     if use_router and use_ca:
    #     # 处理text
    #         combined_text = torch.cat(text_token, dim=0)
    #         if combined_text.dim() == 2:
    #             combined_text = combined_text.unsqueeze(0)

    #         # print("combined_text shape:", combined_text.shape)

    #         # 初始化最终输出
    #         batch_size, text_len, dim = combined_text.shape
    #         batch_size = image_features.shape[0]
    #         combined_features = torch.zeros(
    #             batch_size, text_len, dim, 
    #             device=combined_text.device, 
    #             dtype=combined_text.dtype
    #         )
            
    #         # Use router to select top-5 layers
    #         # For batched version, use first sample's selection for all
    #         top_indices, top_weights, all_probs = self.get_model().layer_router(combined_text)
            
    #         # Use the most common selection or first sample's selection
    #         # selected_indices = top_indices[0]
    #         # selected_weights = top_weights[0]
    #         print(f"Selected layers: {top_indices.tolist()}")
    #         # print(f"Layer weights: {top_weights.tolist()}")
            
    #         for idx in range(len(top_indices)):
    #         # 获取当前层特征 [batch, num_patches, dim]
    #             # layer_feat = image_forward_outs.hidden_states[i]
    #             # layer_feat = image_forward_outs.hidden_states[i].half()
    #             layer_idx = top_indices[idx].item()  # 转为整数用于索引
    #             weight = top_weights[idx]

    #             layer_feat = image_forward_outs.hidden_states[layer_idx][:, 1:].to(image_features.dtype)
    #             # print("layer_feat shape:", layer_feat.shape)
    #             # layer_feat = image_forward_outs.hidden_states[i].to(self.get_model().mm_projector.weight.dtype)
    #             # print("layer_feat type:", type(layer_feat))
    #             layer_features = self.get_model().mm_projector(layer_feat)

    #             attended = self.get_model().ca(combined_text, layer_features)

    #             # print("layer_feat type:", type(layer_feat))
    #             combined_features += weight * attended
                
    #             # print(f"Processed layer {idx}")
    #             # print("combined_features shape:", combined_features.shape)

    #     else:
    #         combined_features = self.get_model().mm_projector(image_features)
            
    #     return combined_features

    def encode_images(self, images, text_token):
        """
        Batched version - supports flexible combinations of router and CA.
        Handles: no router/CA, only router, only CA, or both.
        """
        # 获取24层特征
        image_features, image_forward_outs = self.get_model().get_vision_tower()(images)

        use_router = getattr(self.get_model().config, 'use_router', False)
        use_ca = getattr(self.get_model().config, 'use_ca', False)

        # Case 1: 都不使用 - 直接返回投影后的特征
        if not use_router and not use_ca:
            combined_features = self.get_model().mm_projector(image_features)
            return combined_features

        # 准备text token
        combined_text = torch.cat(text_token, dim=0)
        if combined_text.dim() == 2:
            combined_text = combined_text.unsqueeze(0)
        # print("combined_text shape:",combined_text.shape)

        batch_size = image_features.shape[0]
        text_len = combined_text.shape[1]
        
        # Case 2: 只使用CA，不使用Router - 对最后一层特征做CA
        if use_ca and not use_router:
            # 使用最后一层的特征（layer 23，即index -1）
            last_layer_feat = image_forward_outs.hidden_states[-1][:, 1:].to(image_features.dtype)
            last_layer_features = self.get_model().mm_projector(last_layer_feat)
            combined_features = self.get_model().ca(combined_text, last_layer_features)
            return combined_features

        # Case 3: 只使用Router，不使用CA - 加权融合多层特征
        if use_router and not use_ca:
            dim = self.get_model().mm_projector(image_forward_outs.hidden_states[0][:, 1:]).shape[-1]
            combined_features = torch.zeros(
                batch_size, image_features.shape[1], dim,
                device=image_features.device,
                dtype=image_features.dtype
            )
            
            if self.training:
                router_output = self.get_model().layer_router(combined_text, return_loss=True)
                
                if len(router_output) == 4:
                    top_indices, top_weights, all_probs, diversity_loss = router_output
                    
                    if hasattr(self.get_model(), '_router_diversity_losses'):
                        self.get_model()._router_diversity_losses.append(diversity_loss)
                else:
                    top_indices, top_weights, all_probs = router_output
            else:
                top_indices, top_weights, all_probs = self.get_model().layer_router(
                    combined_text, return_loss=False
                )
            
            # print(f"Selected layers: {top_indices.tolist()}")
            
            # 加权融合选中的层
            for idx in range(len(top_indices)):
                layer_idx = top_indices[idx].item()
                weight = top_weights[idx]
                
                layer_feat = image_forward_outs.hidden_states[layer_idx][:, 1:].to(image_features.dtype)
                layer_features = self.get_model().mm_projector(layer_feat)
                combined_features += weight * layer_features
            
            # print("Combined features shape (Router only):", combined_features.shape)
            return combined_features

        # Case 4: 同时使用Router和CA - 原有逻辑
        if use_router and use_ca:
            dim = combined_text.shape[-1]
            combined_features = torch.zeros(
                batch_size, text_len, dim,
                device=combined_text.device,
                dtype=combined_text.dtype
            )

            # if torch.isnan(combined_features).any():
            #     print("⚠️ combined_features initialized with NaN!")
            
            if self.training:
                router_output = self.get_model().layer_router(combined_text, return_loss=True)
                
                if len(router_output) == 4:
                    top_indices, top_weights, all_probs, diversity_loss = router_output
                    
                    if hasattr(self.get_model(), '_router_diversity_losses'):
                        self.get_model()._router_diversity_losses.append(diversity_loss)
                        # print(f"  ✅ Added diversity_loss: {diversity_loss.item():.6f}")
                else:
                    top_indices, top_weights, all_probs = router_output
            else:
                top_indices, top_weights, all_probs = self.get_model().layer_router(
                    combined_text, return_loss=False
                )
            
            # print(f"Selected layers: {top_indices.tolist()}")
            
            # Router选中的层 + CA attention
            for idx in range(len(top_indices)):
                layer_idx = top_indices[idx].item()
                weight = top_weights[idx]

                layer_feat = image_forward_outs.hidden_states[layer_idx][:, 1:].to(image_features.dtype)
                # print("layer_feat shape",layer_feat.shape)

                # if torch.isnan(layer_feat).any() or torch.isinf(layer_feat).any():
                #     print(f"⚠️ Layer {layer_idx}: NaN/Inf in vision tower output")

                layer_features = self.get_model().mm_projector(layer_feat)
                # print("layer_features shape",layer_features.shape)

                if torch.isnan(layer_features).any() or torch.isinf(layer_features).any():
                    print(f"⚠️ Layer {layer_idx}: NaN/Inf in projector output")
                    print(f"   Range: [{layer_features.min():.4f}, {layer_features.max():.4f}]")

                # print(f"\n🔍 Before CA for layer {layer_idx}:")
                # print(f"  layer_features: range=[{layer_features.min():.4f}, {layer_features.max():.4f}], has_inf={torch.isinf(layer_features).any()}")
                attended = self.get_model().ca(combined_text, layer_features)

                if torch.isnan(attended).any() or torch.isinf(attended).any():
                    print(f"⚠️ Layer {layer_idx}: NaN/Inf in CA output")
                    print(f"   Range: [{attended.min():.4f}, {attended.max():.4f}]")
                
                combined_features += weight * attended
                # print("combined_features shape:",combined_features.shape)
                # print(f"  combined_features: range=[{combined_features.min():.4f}, {combined_features.max():.4f}], has_inf={torch.isinf(combined_features).any()}")
            
            return combined_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, CA=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if self.training:
            tune_router = getattr(self.get_model().config, 'tune_router', False)
            if tune_router and hasattr(self.get_model(), 'layer_router'):
                self.get_model()._router_diversity_losses = []
        
        # ################################
        origi_input_ids = input_ids
        origi_labels = labels
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        no_attention_mask = False
        no_position_ids = False
        no_labels =False

        if attention_mask is None:
            no_attention_mask = True
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            no_position_ids = True
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            no_labels = True
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        if self.training:  # 训练阶段
            input_ids = origi_input_ids
            labels = origi_labels

            if no_attention_mask:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if no_position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if no_labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)
        else:  # 评估阶段
            pass

        # ################################
        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, cur_input_embeds_no_im)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, cur_input_embeds_no_im)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.

        # remove the padding using attention_mask -- FIXME
        if self.training:
            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)
        else:
            pass

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)


            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # print("new_input_embeds shape", new_input_embeds.shape)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False