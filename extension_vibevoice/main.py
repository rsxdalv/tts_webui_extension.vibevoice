import gradio as gr


def vibevoice_ui():
    gr.Markdown(
        """
    # Vibevoice
    
    This is a template extension. Replace this content with your extension's functionality.
    
    To use it, simply modify this UI and add your custom logic.
    """
    )
    
    # Add your UI components here
    # Example:
    # with gr.Row():
    #     with gr.Column():
    #         input_text = gr.Textbox(label="Input")
    #         button = gr.Button("Process")
    #     with gr.Column():
    #         output_text = gr.Textbox(label="Output")
    # 
    # button.click(
    #     fn=your_processing_function,
    #     inputs=[input_text],
    #     outputs=[output_text],
    #     api_name="vibevoice",
    # )


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
