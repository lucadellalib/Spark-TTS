# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparktts.modules.encoder_decoder.feat_decoder import Decoder
from sparktts.modules.encoder_decoder.feat_encoder import Encoder
from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
from sparktts.modules.vq.factorized_vector_quantize import \
    FactorizedVectorQuantize
from sparktts.utils.file import load_config


class BiCodec(nn.Module):
    """
    BiCodec model for speech synthesis, incorporating a speaker encoder, feature encoder/decoder,
    quantizer, and wave generator.
    """

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        **kwargs,
    ) -> None:
        """
        Initializes the BiCodec model with the required components.

        Args:
            mel_params (dict): Parameters for the mel-spectrogram transformer.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            quantizer (nn.Module): Quantizer module.
            speaker_encoder (nn.Module): Speaker encoder module.
            prenet (nn.Module): Prenet network.
            postnet (nn.Module): Postnet network.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_checkpoint(
        cls, model_dir="SparkAudio/Spark-TTS-0.5B", **kwargs
    ) -> "BiCodec":
        """
        Loads the model from a checkpoint.

        Args:
            model_dir (Path): Path to the model directory containing checkpoint and config.

        Returns:
            BiCodec: The initialized BiCodec model.
        """
        import os

        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_dir)
        codec_dir = os.path.join(model_dir, "BiCodec")
        ckpt_path = f"{codec_dir}/model.safetensors"
        config = load_config(f"{codec_dir}/config.yaml")["audio_tokenizer"]
        mel_params = config["mel_params"]
        encoder = Encoder(**config["encoder"])
        quantizer = FactorizedVectorQuantize(**config["quantizer"])
        prenet = Decoder(**config["prenet"])
        postnet = Decoder(**config["postnet"])
        decoder = WaveGenerator(**config["decoder"])
        speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
        )

        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        #for key in missing_keys:
        #    print(f"Missing tensor: {key}")
        #for key in unexpected_keys:
        #    print(f"Unexpected tensor: {key}")

        model.remove_weight_norm()

        model.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{model_dir}/wav2vec2-large-xlsr-53"
        )
        model.feature_extractor = Wav2Vec2Model.from_pretrained(
            f"{model_dir}/wav2vec2-large-xlsr-53"
        )
        model.feature_extractor.config.output_hidden_states = True

        return model.eval()

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        inputs = self.processor(
            [x.numpy() for x in wavs[:, 0].cpu()],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    @torch.no_grad()
    def tokenize(self, wav):
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = self.extract_wav2vec2_features(wav)
        mel = self.mel_transformer(wav).squeeze(1)
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))
        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """
        Detokenizes the semantic and global tokens into a waveform.

        Args:
            semantic_tokens (tensor): Semantic tokens.
            global_tokens (tensor): Global tokens.

        Returns:
            tensor: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Removes weight normalization from all layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                pass  # The module didn't have weight norm

        self.apply(_remove_weight_norm)


# Test the model
if __name__ == "__main__":
    model = BiCodec.load_from_checkpoint().cuda()
    print(sum([x.numel() for x in model.parameters()]) / 1e6)

    # Generate random inputs for testing
    duration = 1
    x = torch.randn(2, 1, int(duration * 16000)).cuda()

    # Forward pass
    semantic_tokens, global_tokens = model.tokenize(x)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)
    print(wav_recon.shape)
