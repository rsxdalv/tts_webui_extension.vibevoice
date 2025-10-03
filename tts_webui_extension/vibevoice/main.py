import numpy as np
import gradio as gr
import torch


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .backend_api import VibeVoiceDemo


def create_demo_interface(_instance: "VibeVoiceDemo"):
    gr.HTML(
        """
    <div class="main-header">
        <h1>🎙️ Vibe Podcasting </h1>
        <p>Generating Long-form Multi-speaker AI Podcast with VibeVoice</p>
    </div>
    """
    )

    with gr.Row():
        # Left column - Settings
        with gr.Column(scale=1, elem_classes="settings-card"):
            gr.Markdown("### 🎛️ **Podcast Settings**")

            # Number of speakers
            num_speakers = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=1,
                label="Number of Speakers",
                elem_classes="slider-container",
            )

            # Speaker selection
            gr.Markdown("### 🎭 **Speaker Selection**")

            available_speaker_names = list(_instance.available_voices.keys())
            # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
            default_speakers = [
                "en-Alice_woman",
                "en-Carter_man",
                "en-Frank_man",
                "en-Maya_woman",
            ]

            speaker_selections = []
            for i in range(4):
                default_value = (
                    default_speakers[i] if i < len(default_speakers) else None
                )
                speaker = gr.Dropdown(
                    choices=available_speaker_names,
                    value=default_value,
                    label=f"Speaker {i+1}",
                    visible=(i < 2),  # Initially show only first 2 speakers
                    elem_classes="speaker-item",
                )
                speaker_selections.append(speaker)

            # Advanced settings
            gr.Markdown("### ⚙️ **Advanced Settings**")

            # Sampling parameters (contains all generation settings)
            with gr.Accordion("Generation Parameters", open=False):
                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.3,
                    step=0.05,
                    label="CFG Scale (Guidance Strength)",
                    # info="Higher values increase adherence to text",
                )

                inference_steps = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Inference Steps",
                )

                attn_implementation = gr.Dropdown(
                    choices=["sdpa", "flash_attention_2"],
                    value="flash_attention_2",
                    label="Attention Implementation",
                    visible=False,
                )

            gr.Markdown("### 🎵 **Download voices (required, restart after download)**")
            gr.Markdown(
                "Voices are located in ./voices/vibevoice/ and scanned on startup."
            )

            download_btn = gr.Button("Download Voices")

            def download_voices():
                import os
                import requests

                yield "Downloading..."
                for voice in [
                    "en-Alice_woman",
                    "en-Carter_man",
                    "en-Frank_man",
                    "en-Maya_woman",
                    "in-Samuel_man",
                    "zh-Anchen_man_bgm",
                    "zh-Bowen_man",
                    "zh-Xinran_woman",
                ]:
                    yield f"Downloading {voice}..."
                    voice_url = f"https://raw.githubusercontent.com/rsxdalv/VibeVoice/refs/heads/main/demo/voices/{voice}.wav"
                    voice_path = f"./voices/vibevoice/{voice}.wav"
                    os.makedirs(os.path.dirname(voice_path), exist_ok=True)

                    response = requests.get(voice_url)
                    if response.status_code == 200:
                        with open(voice_path, "wb") as f:
                            f.write(response.content)
                    yield f"Downloaded {voice}"

            download_btn.click(fn=download_voices, inputs=[], outputs=[download_btn])

        # Right column - Generation
        with gr.Column(scale=2, elem_classes="generation-card"):
            gr.Markdown("### 📝 **Script Input**")

            script_input = gr.Textbox(
                label="Conversation Script",
                placeholder="""Enter your podcast script here. You can format it as:

Speaker 0: Welcome to our podcast today!
Speaker 1: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                lines=12,
                max_lines=20,
                elem_classes="script-input",
            )

            # Button row with Random Example on the left and Generate on the right
            with gr.Row():
                # Random example button (now on the left)
                random_example_btn = gr.Button(
                    "🎲 Random Example",
                    size="lg",
                    variant="secondary",
                    elem_classes="random-btn",
                    scale=1,  # Smaller width
                )

                # Generate button (now on the right)
                generate_btn = gr.Button(
                    "🚀 Generate Podcast",
                    size="lg",
                    variant="primary",
                    elem_classes="generate-btn",
                    scale=2,  # Wider than random button
                )

            # Stop button
            stop_btn = gr.Button(
                "🛑 Stop Generation",
                size="lg",
                variant="stop",
                elem_classes="stop-btn",
                visible=False,
            )

            # Streaming status indicator
            streaming_status = gr.HTML(
                value="""
                <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                            border: 1px solid rgba(34, 197, 94, 0.3); 
                            border-radius: 8px; 
                            padding: 0.75rem; 
                            margin: 0.5rem 0;
                            text-align: center;
                            font-size: 0.9rem;
                            color: #166534;">
                    <span class="streaming-indicator"></span>
                    <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                </div>
                """,
                visible=False,
                elem_id="streaming-status",
            )

            # Output section
            gr.Markdown("### 🎵 **Generated Podcast**")

            # Streaming audio output (outside of tabs for simpler handling)
            audio_output = gr.Audio(
                label="Streaming Audio (Real-time)",
                type="numpy",
                elem_classes="audio-output",
                streaming=True,  # Enable streaming mode
                autoplay=True,
                show_download_button=False,  # Explicitly show download button
                visible=True,
            )

            # Complete audio output (non-streaming)
            complete_audio_output = gr.Audio(
                label="Complete Podcast (Download after generation)",
                type="numpy",
                elem_classes="audio-output complete-audio-section",
                streaming=False,  # Non-streaming mode
                autoplay=False,
                show_download_button=True,  # Explicitly show download button
                visible=False,  # Initially hidden, shown when audio is ready
            )

            gr.Markdown(
                """
            *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
            *💡 **Complete Audio**: Will appear below after generation finishes*
            """
            )

            # Generation log
            log_output = gr.Textbox(
                label="Generation Log",
                lines=8,
                max_lines=15,
                interactive=False,
                elem_classes="log-output",
            )

    def update_speaker_visibility(num_speakers):
        updates = []
        for i in range(4):
            updates.append(gr.update(visible=(i < num_speakers)))
        return updates

    num_speakers.change(
        fn=update_speaker_visibility,
        inputs=[num_speakers],
        outputs=speaker_selections,
    )

    # Main generation function with streaming
    def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
        """Wrapper function to handle the streaming generation call."""
        try:
            # Extract speakers and parameters
            speakers = speakers_and_params[:4]  # First 4 are speaker selections
            cfg_scale = speakers_and_params[4]  # CFG scale
            inference_steps = speakers_and_params[5]  # Inference steps

            # Clear outputs and reset visibility at start
            yield None, gr.update(
                value=None, visible=False
            ), "🎙️ Starting generation...", gr.update(visible=True), gr.update(
                visible=False
            ), gr.update(
                visible=True
            )

            # The generator will yield multiple times
            final_log = "Starting generation..."

            for (
                streaming_audio,
                complete_audio,
                log,
                streaming_visible,
            ) in _instance.generate_podcast_streaming(
                num_speakers=int(num_speakers),
                script=script,
                speaker_1=speakers[0],
                speaker_2=speakers[1],
                speaker_3=speakers[2],
                speaker_4=speakers[3],
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
            ):
                final_log = log

                # Check if we have complete audio (final yield)
                if complete_audio is not None:
                    # Final state: clear streaming, show complete audio
                    yield None, gr.update(
                        value=complete_audio, visible=True
                    ), log, gr.update(visible=False), gr.update(
                        visible=True
                    ), gr.update(
                        visible=False
                    )
                else:
                    # Streaming state: update streaming audio only
                    if streaming_audio is not None:
                        yield streaming_audio, gr.update(
                            visible=False
                        ), log, streaming_visible, gr.update(visible=False), gr.update(
                            visible=True
                        )
                    else:
                        # No new audio, just update status
                        yield None, gr.update(
                            visible=False
                        ), log, streaming_visible, gr.update(visible=False), gr.update(
                            visible=True
                        )

        except Exception as e:
            error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            # Reset button states on error
            yield None, gr.update(value=None, visible=False), error_msg, gr.update(
                visible=False
            ), gr.update(visible=True), gr.update(visible=False)

    def stop_generation_handler():
        """Handle stopping generation."""
        _instance.stop_audio_generation()
        # Return values for: log_output, streaming_status, generate_btn, stop_btn
        return (
            "🛑 Generation stopped.",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    # Add a clear audio function
    def clear_audio_outputs():
        """Clear both audio outputs before starting new generation."""
        return None, gr.update(value=None, visible=False)

    # Connect generation button with streaming outputs
    generate_btn.click(
        fn=clear_audio_outputs,
        inputs=[],
        outputs=[audio_output, complete_audio_output],
        queue=False,
    ).then(
        fn=generate_podcast_wrapper,
        inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale] + [inference_steps],
        outputs=[
            audio_output,
            complete_audio_output,
            log_output,
            streaming_status,
            generate_btn,
            stop_btn,
        ],
        queue=True,  # Enable Gradio's built-in queue
    )

    # Connect stop button
    stop_btn.click(
        fn=stop_generation_handler,
        inputs=[],
        outputs=[log_output, streaming_status, generate_btn, stop_btn],
        queue=False,  # Don't queue stop requests
    ).then(
        # Clear both audio outputs after stopping
        fn=lambda: (None, None),
        inputs=[],
        outputs=[audio_output, complete_audio_output],
        queue=False,
    )

    # Function to randomly select an example
    def load_random_example():
        """Randomly select and load an example script."""
        import random

        # Get available examples
        if hasattr(_instance, "example_scripts") and _instance.example_scripts:
            example_scripts = _instance.example_scripts
        else:
            # Fallback to default
            example_scripts = [
                [
                    2,
                    "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!",
                ]
            ]

        # Randomly select one
        if example_scripts:
            selected = random.choice(example_scripts)
            num_speakers_value = selected[0]
            script_value = selected[1]

            # Return the values to update the UI
            return num_speakers_value, script_value

        # Default values if no examples
        return 2, ""

    # Connect random example button
    random_example_btn.click(
        fn=load_random_example,
        inputs=[],
        outputs=[num_speakers, script_input],
        queue=False,  # Don't queue this simple operation
    )

    # Add usage tips
    gr.Markdown(
        """
    ### 💡 **Usage Tips**
    
    - Click **🚀 Generate Podcast** to start audio generation
    - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
    - **Complete Audio** tab provides the full, uninterrupted podcast after generation
    - During generation, you can click **🛑 Stop Generation** to interrupt the process
    - The streaming indicator shows real-time generation progress
    """
    )

    # Add example scripts
    gr.Markdown("### 📚 **Example Scripts**")

    # Use dynamically loaded examples if available, otherwise provide a default
    if hasattr(_instance, "example_scripts") and _instance.example_scripts:
        example_scripts = _instance.example_scripts
    else:
        # Fallback to a simple default example if no scripts loaded
        example_scripts = [
            [
                1,
                "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech.",
            ]
        ]

    gr.Examples(
        examples=example_scripts,
        inputs=[num_speakers, script_input],
        label="Try these example scripts:",
    )


def vibevoice_ui():
    """Main function to run the demo."""

    # Initialize demo instance
    from .backend_api import get_instance
    from .backend_api import get_instance_15

    with gr.Tabs():
        with gr.Tab("VibeVoice-Large"):
            demo_instance_2 = get_instance("rsxdalv/VibeVoice-Large")

            create_demo_interface(demo_instance_2)
        with gr.Tab("VibeVoice-1.5B"):
            demo_instance = get_instance_15("microsoft/VibeVoice-1.5B")

            create_demo_interface(demo_instance)


def extension__tts_generation_webui():
    vibevoice_ui()

    return {
        "package_name": "extension_vibevoice",
        "name": "Vibevoice",
        "requirements": "git+https://github.com/yourusername/extension_vibevoice@main",
        "description": "A template extension for TTS Generation WebUI",
        "extension_type": "interface",
        "extension_class": "tools",
        "author": "Your Name",
        "extension_author": "Your Name",
        "license": "MIT",
        "website": "https://github.com/yourusername/extension_vibevoice",
        "extension_website": "https://github.com/yourusername/extension_vibevoice",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        with gr.Tab("Vibevoice", id="vibevoice"):
            vibevoice_ui()

    demo.launch(
        server_port=7772,  # Change this port if needed
    )
