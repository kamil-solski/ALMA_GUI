import gradio as gr
from model import ModelHandler, Translator

# Instantiate model and translator
model_handler = ModelHandler()
translator = Translator(model_handler)

def translate_text(user_input):
    """Run translation inference on user-provided text."""
    chunks = translator.split_text_into_chunks(user_input)
    output = translator.translate(chunks)
    return output

# Create Gradio interface
description = "Tłumacz tekst bez limitów."
with gr.Blocks(title="Alternatywa dla DeepL") as app:
    gr.Markdown("# Translation Interface")
    gr.Markdown(description)

    with gr.Row():
        input_box = gr.Textbox(label="Input Text (Polish)", lines=10, placeholder="Enter text to translate...")
        output_box = gr.Textbox(label="Translated Text (English)", lines=10, interactive=False)
    
    translate_btn = gr.Button("Translate")
    translate_btn.click(fn=translate_text, inputs=[input_box], outputs=[output_box])

# Launch the Gradio interface
app.launch()
