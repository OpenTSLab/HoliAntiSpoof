from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

from .modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration, QWEN2_5OMNITHINKER_INPUTS_DOCSTRING,
    Qwen2_5OmniThinkerCausalLMOutputWithPast
)
from .configuration_qwen2_5_omni import Qwen2_5OmniThinkerWithSpoofConfig


class Qwen2_5OmniThinkerWithSpoofForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    def build_model(self, config: Qwen2_5OmniThinkerWithSpoofConfig):
        super().build_model(config)
        self.spoof_proj = nn.Sequential(
            nn.Linear(config.spoof_embed_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniThinkerCausalLMOutputWithPast, config_class="Qwen2_5OmniThinkerConfig"
    )
    def forward(
        self,
        spoof_embeds: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        spoof_embeds = self.spoof_proj(spoof_embeds.to(self.dtype))

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None or (cache_position is not None and cache_position[0] == 0) or
                self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text , audios , image and video
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if input_features is not None:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                audio_outputs = self.audio_tower(
                    input_features,
                    feature_lens=feature_lens,
                    aftercnn_lens=audio_feat_lengths,
                )
                audio_features = audio_outputs.last_hidden_state
                if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")

                audio_mask = ((input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(
                    inputs_embeds.device
                ))
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                audio_features = self.insert_spoof_embedding(audio_features, audio_output_lengths, spoof_embeds)

                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = ((input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(
                    inputs_embeds.device
                ))
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = ((input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(
                    inputs_embeds.device
                ))
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # if torch.cuda.current_device() == 0:
                #     print(f"RANK 0 video_embeds: {video_embeds.shape}")
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # if (input_features is not None or pixel_values_videos
            #     is not None) and torch.cuda.current_device() == 0:
            #     print(f"RANK 0 inputs_embeds: {inputs_embeds.shape}")

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        loss = None
        logits = self.lm_head(hidden_states)
        if labels is not None:
            loss = LigerForCausalLMLoss(
                hidden_states=hidden_states,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                hidden_size=hidden_states.size(-1)
            )

        if not return_dict:
            output = (logits, ) + outputs
            return (loss, ) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def insert_spoof_embedding(
        self, audio_features: torch.Tensor, audio_feature_lengths: torch.Tensor, spoof_embeds: torch.Tensor
    ):
        """
        audio_features: (T, E)
        lengths: list or 1D tensor of segment lengths [T1, T2, ..., TN]
        spoof_embed: (N, E)
        
        return: (T + N, E)
        """
        T, E = audio_features.shape
        N = len(audio_feature_lengths)

        end_idx = torch.cumsum(audio_feature_lengths, dim=0)  # [T1, T1+T2, ..., T1+T2+...+TN]
        insert_positions = end_idx + torch.arange(N, device=audio_features.device)
        # insert_positions: [T1, T1+T2+1, ..., T1+T2+...+TN+N-1]

        out = torch.zeros((T + N, E), device=audio_features.device, dtype=audio_features.dtype)

        offset = torch.arange(T, device=audio_features.device)
        offset_adjust = torch.repeat_interleave(torch.arange(N, device=audio_features.device), audio_feature_lengths)
        # offset_adjust: [0, 0, ..., 0 (T1 x 0), 1, 1, ..., 1 (T2 x 1), ..., N-1, N-1, ..., N-1 (TN x N-1)]
        offset += offset_adjust
        # offset: [0, 1, ..., T1-1, (len=T1) T1+1, T1+2, ..., T1+T2 (len=T2), ..., ]
        # each segment is offset by the segment index

        out[offset] = audio_features
        out[insert_positions] = spoof_embeds
        return out
