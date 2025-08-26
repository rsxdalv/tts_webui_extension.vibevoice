"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

__file__ = "."


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        # self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(
            self.model_path,
        )

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        if hasattr(self.model.model, "language_model"):
            print(
                f"Language model attention: {self.model.model.language_model.config._attn_implementation}"
            )

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join("voices", "vibevoice")

        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        # Scan for all WAV files in the voices directory
        self.voice_presets = {}

        # Get all .wav files in the voices directory
        wav_files = [
            f
            for f in os.listdir(voices_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))
            and os.path.isfile(os.path.join(voices_dir, f))
        ]

        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path
            for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        if not self.available_voices:
            raise gr.Error(
                "No voice presets found. Please add .wav files to the demo/voices directory."
            )

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def generate_podcast_streaming(
        self,
        num_speakers: int,
        script: str,
        speaker_1: str = None,
        speaker_2: str = None,
        speaker_3: str = None,
        speaker_4: str = None,
        cfg_scale: float = 1.3,
    ) -> Iterator[tuple]:
        try:
            if not hasattr(self, "model") or self.model is None:
                self.load_model()

            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True

            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][
                :num_speakers
            ]

            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(
                        f"Error: Please select a valid speaker for Speaker {i+1}."
                    )

            # Build initial log
            log = f"🎙️ Generating podcast with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"

            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)

            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"

            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Parse script to assign speaker ID's
            lines = script.strip().split("\n")
            formatted_script_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line already has speaker format
                if line.startswith("Speaker ") and ":" in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

            formatted_script = "\n".join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"

            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            start_time = time.time()

            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Create audio streamer
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)

            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer

            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer),
            )
            generation_thread.start()

            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(
                    timeout=5.0
                )  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15  # Yield every 15 seconds
            min_chunk_size = sample_rate * 30  # At least 2 seconds of audio

            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)

            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all

            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break

                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk

                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)

                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()

                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)

                # Store for final statistics
                all_audio_chunks.append(audio_16bit)

                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)

                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time

                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (
                    pending_audio_size >= min_chunk_size
                    or time_since_last_yield >= min_yield_interval
                ):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True

                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = (
                        sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    )

                    log_update = (
                        log
                        + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    )

                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(
                        visible=True
                    )

                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time

            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = (
                    sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                )
                log_update = (
                    log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                )
                yield (sample_rate, final_new_audio), None, log_update, gr.update(
                    visible=True
                )
                has_yielded_audio = True  # Mark that we yielded audio

            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False

            generation_time = time.time() - start_time

            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(
                    visible=False
                )
                return

            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")

            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate

                final_log = (
                    log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                )
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."

                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(
                    visible=False
                )
                return

            if not has_received_chunks:
                error_log = (
                    log
                    + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                )
                yield None, None, error_log, gr.update(visible=False)
                return

            if not has_yielded_audio:
                error_log = (
                    log
                    + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                )
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate

                final_log = (
                    log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                )
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."

                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(
                    visible=False
                )
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)

        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)

    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            if not hasattr(self, "model") or self.model is None:
                self.load_model()

            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return

            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )

        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()

    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")

    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []

        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return

        # Get all .txt files in the text_examples directory
        txt_files = sorted(
            [
                f
                for f in os.listdir(examples_dir)
                if f.lower().endswith(".txt")
                and os.path.isfile(os.path.join(examples_dir, f))
            ]
        )

        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)

            import re

            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r"(\d+)min", txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(
                        f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit"
                    )
                    continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    script_content = f.read().strip()

                # Remove empty lines and lines with only whitespace
                script_content = "\n".join(
                    line for line in script_content.split("\n") if line.strip()
                )

                if not script_content:
                    continue

                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)

                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")

            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")

        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")

    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re

        speakers = set()

        lines = script.strip().split("\n")
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r"^Speaker\s+(\d+)\s*:", line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)

        # If no speakers found, default to 1
        if not speakers:
            return 1

        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)

        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))

    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


demo_instance = None


def get_instance():
    global demo_instance
    if demo_instance is None:
        """Get the VibeVoice demo instance."""
        demo_instance = VibeVoiceDemo(
            model_path="microsoft/VibeVoice-1.5B",
            device="cuda" if torch.cuda.is_available() else "cpu",
            inference_steps=10,
        )

    return demo_instance
