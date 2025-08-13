"""
PyTorch Deformable DETR model with Dual Memory modifications.

This file contains the experimental classes that inherit from the base
Deformable DETR implementation and add the dual-memory architecture.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the base classes from the original model file
from .modeling_deformable_detr import (
    DeformableDetrModel,
    DeformableDetrForObjectDetection,
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrMultiheadAttention,
    DeformableDetrConfig,
    DeformableDetrDecoderOutput,
    DeformableDetrModelOutput,
    inverse_sigmoid,
    _prepare_4d_attention_mask,
)

# Import the memory module we will be instantiating
from .memory.proposal_query_memory import ProposalQueryMemory

class DualMemoryMultiheadAttention(DeformableDetrMultiheadAttention):
    """
    An extension of the Deformable DETR Multihead Attention that handles two
    separate prompt streams and multiple integration strategies.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        prompts_all=None,
        prompts_specific=None,
        prefix_tuning: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, target_len, embed_dim = hidden_states.size()
        
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            qkv_base = self.with_pos_embed(hidden_states, position_embeddings)
        else:
            hidden_states_original = hidden_states
            qkv_base = hidden_states

        q_to_ek_strategy = self.config.q_to_ek_strategy

        query_states = self.q_proj(qkv_base) * self.scaling
        key_states = self.k_proj(qkv_base)
        value_states = self.v_proj(hidden_states_original)

        if prompts_specific is not None:
            pk_specific, pv_specific = prompts_specific
            
            if q_to_ek_strategy == "query_bias":
                if pv_specific.shape[1] == target_len:
                    query_states = query_states + pv_specific
            elif q_to_ek_strategy == "pos_embed_prefix":
                if position_embeddings is not None:
                    pk_specific = pk_specific + position_embeddings
                key_states = torch.cat((pk_specific, key_states), dim=1)
                value_states = torch.cat((pv_specific, value_states), dim=1)
            elif q_to_ek_strategy == "masked_prefix":
                key_states = torch.cat((pk_specific, key_states), dim=1)
                value_states = torch.cat((pv_specific, value_states), dim=1)

        if prompts_all is not None and prefix_tuning:
            pk_global, pv_global = prompts_all
            key_states = torch.cat((pk_global, key_states), dim=1)
            value_states = torch.cat((pv_global, value_states), dim=1)
        
        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        if q_to_ek_strategy == "masked_prefix" and prompts_specific is not None:
            num_specific_prompts = prompts_specific[0].shape[1]
            l_per_query = num_specific_prompts // target_len
            num_all_prompts = prompts_all[0].shape[1] if prompts_all is not None else 0
            
            mask2d = torch.zeros((target_len, source_len), device=attn_weights.device, dtype=torch.bool)
            mask2d[:, :num_all_prompts] = True
            mask2d[:, -target_len:] = True

            for i in range(target_len):
                start = num_all_prompts + i * l_per_query
                end = start + l_per_query
                mask2d[i, start:end] = True

            neg_inf = -1e9
            float_mask = torch.full((target_len, source_len), neg_inf, device=attn_weights.device, dtype=attn_weights.dtype)
            float_mask[mask2d] = 0.0
            mask4d = float_mask.unsqueeze(0).unsqueeze(0)
            
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights + mask4d
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size * self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        if q_to_ek_strategy == "output_bias" and prompts_specific is not None:
            _, pv_specific = prompts_specific
            if pv_specific.shape[1] == target_len:
                attn_output = attn_output + pv_specific

        return attn_output, attn_weights_reshaped


class DualMemoryDecoderLayer(DeformableDetrDecoderLayer):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.self_attn = DualMemoryMultiheadAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        prompts_all=None,
        prompts_specific=None,
    ):
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            prompts_all=prompts_all,
            prompts_specific=prompts_specific,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs


class DualMemoryDecoder(DeformableDetrDecoder):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([DualMemoryDecoderLayer(config) for _ in range(config.decoder_layers)])

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        query=None,
        task_id=None,
        train=False,
        prompts_all=None,
        prompts_specific=None,
        class_labels=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            prompts_all_to_pass, prompts_specific_to_pass = None, None
            strategy = self.config.dual_memory_strategy
            switch_layer = self.config.dual_memory_switch_layer

            should_generate_all = (strategy == "hybrid_everywhere") or \
                                  (strategy == "phased_global_specific" and idx < switch_layer) or \
                                  (strategy == "phased_hybrid_specific" and idx < switch_layer)

            should_generate_specific = (strategy == "hybrid_everywhere") or \
                                       (strategy == "phased_hybrid_specific" and idx < switch_layer) or \
                                       (strategy == "phased_global_specific" and idx >= switch_layer) or \
                                       (strategy == "phased_hybrid_specific" and idx >= switch_layer)

            if query is not None:
                if should_generate_all and prompts_all is not None:
                    prompts_all_to_pass, _, _ = prompts_all.forward(
                        query, idx, hidden_states, train=train, task_id=task_id
                    )
                if should_generate_specific and prompts_specific is not None:
                    prompts_specific_to_pass, _, _ = prompts_specific.forward(
                        query, idx, hidden_states, train=train, task_id=task_id
                    )

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                prompts_all=prompts_all_to_pass,
                prompts_specific=prompts_specific_to_pass,
            )

            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class DualMemoryDetrModel(DeformableDetrModel):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.decoder = DualMemoryDecoder(config)
        
        if config.use_prompts:
            self.prompts_all = ProposalQueryMemory(
                emb_d=config.d_model, key_d=config.d_model, default_units=10, 
                e_p_length=2, local_query=config.local_query
            )
            self.prompts_q_to_ek = ProposalQueryMemory(
                emb_d=config.d_model, key_d=config.d_model, default_units=10, 
                e_p_length=2, local_query=config.local_query
            )
            self.prompts = self.prompts_all
        else:
            self.prompts = None
            self.prompts_all = None
            self.prompts_q_to_ek = None

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query=None,
        train=False,
        task_id=0,
        class_labels=None,
    ) -> Union[Tuple[torch.FloatTensor], DeformableDetrModelOutput]:
        
        # This is a full copy of the original DeformableDetrModel.forward, with one change:
        # It passes prompts_all and prompts_q_to_ek to the decoder.
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        sources, masks = [], []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        query_embeds = None
        if not self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        source_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            bs, c, h, w = source.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        batch_size, _, num_channels = encoder_outputs[0].shape
        
        query_embed, target = torch.split(query_embeds, num_channels, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_id=task_id,
            prompts_all=self.prompts_all,
            prompts_specific=self.prompts_q_to_ek,
            train=train,
            query=query,
            class_labels=class_labels,
        )

        return DeformableDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class DualMemoryDetrForObjectDetection(DeformableDetrForObjectDetection):
    def __init__(self, config: DeformableDetrConfig, default=True, log_file=None):
        super().__init__(config, default=default, log_file=log_file)
        self.model = DualMemoryDetrModel(config)
