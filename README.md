The main purpose of this repo is to provide simple GUI for X-ALMA translator. Better quantification and distillation of the model will be added in future commits.
For now.

X-ALMA is not just translator but normal LLM to talk to. If model achieve such good translations as authors claims (COMET-22). My goal would be to extract only ability for translation. That were model distilation come to play. 
The biggest drawback of the original model is that it takes quite a long time to inference. This may be because it is large and also takes into account some of the user's text analysis, not just translation. The model not only translates, but is also able to answer questions posed to it in the same language.

Although distilation from model from one model to model with different architecture is hard it is possible:
https://arxiv.org/html/2501.16273v1

In my project Model directory uses soft link. Downlaoded models are stored on a external drive.

# CHANGES v1.0.0:
* initial quantization that allow for inference of the X-ALMA model on the RTX3090 (with 6GB of spare memory space)
* basic inference of provided text, docx or pdf files.

# CHANGES v1.0.1:
* fixed how data is loaded to model (longer documents - no truncation).
* added function to check distribution of number of tokens per sentence.

# CHANGES v1.1.0:
* added GUI using gradio
* rebuilded scripts to working program py
* removed support for docx