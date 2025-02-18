import gradio as gr

# Dummy function to simulate file upload handling
def handle_upload(files):
    if files:
        # Simulate processing the uploaded file
        return "File uploaded successfully!", gr.Button.update(value="Clear", variant="stop")
    else:
        return "No file uploaded.", gr.Button.update(value="Upload File", variant="primary")

# Function to clear the uploaded file
def clear_upload():
    return None, gr.Button.update(value="Upload File", variant="primary")

with gr.Blocks() as demo:
    with gr.Row():
        # Input text box
        text_input = gr.Textbox(label="Enter text here", scale=3)
        
        # Upload button
        upload_button = gr.UploadButton("Upload File", file_types=[".pdf", ".docx", ".csv"], file_count="single", scale=1)
        
        # Output display
        output_display = gr.Textbox(label="Output")

    # Handle file upload
    upload_button.upload(handle_upload, inputs=upload_button, outputs=[output_display, upload_button])

    # Handle clear action
    upload_button.click(clear_upload, outputs=[output_display, upload_button])

demo.launch()
