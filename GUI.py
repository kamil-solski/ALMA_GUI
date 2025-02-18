import gradio as gr
from model import ModelHandler, PDFProcessor, Translator
import pathlib

# Instantiate model and translator
model_handler = ModelHandler()
translator = Translator(model_handler)

def process_input(user_input, pdf_file):
    """Use PDF file if provided; otherwise, use direct text input."""
    if pdf_file is not None:
        pdf_path = pathlib.Path(pdf_file.name)
        metadata_path = pdf_path.with_suffix('.json')
        pdf_text = PDFProcessor.extract_text(pdf_path, metadata_path)
        chunks = translator.split_text_into_chunks(pdf_text)
    else:
        chunks = translator.split_text_into_chunks(user_input)
    return translator.translate(chunks)

# Create Gradio interface
description = "Simple text translation without word limits based on X-ALMA model."
with gr.Blocks(title="Open-source translation") as app:
    gr.Markdown("# Open-source translation")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Input Text (Polish)", lines=10, placeholder="Enter text to translate...")
            pdf_file = gr.File(label="Upload PDF")
        
        output_box = gr.Textbox(label="Translated Text (English)", lines=10, interactive=False)
    
    translate_btn = gr.Button("Translate")
    translate_btn.click(fn=process_input, inputs=[input_box, pdf_file], outputs=[output_box])

# Launch the Gradio interface
app.launch()
