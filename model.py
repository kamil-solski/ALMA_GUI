import torch
import pathlib
import pdfplumber
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pathlib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelHandler:
    def __init__(self):
        self.cache_dir = pathlib.Path.cwd() / "Model"
        self.GROUP2LANG = {
            1: ["da", "nl", "de", "is", "no", "sv", "af"],
            2: ["ca", "ro", "gl", "it", "pt", "es"],
            3: ["bg", "mk", "sr", "uk", "ru"],
            4: ["id", "ms", "th", "vi", "mg", "fr"],
            5: ["hu", "el", "cs", "pl", "lt", "lv"],
            6: ["ka", "zh", "ja", "ko", "fi", "et"],
            7: ["gu", "hi", "mr", "ne", "ur"],
            8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
        }
        self.LANG2GROUP = {lang: group for group, langs in self.GROUP2LANG.items() for lang in langs}
        group_id = self.LANG2GROUP["pl"]
        self.model_name = f"haoranxu/X-ALMA-13B-Group{group_id}"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            cache_dir=self.cache_dir
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", cache_dir=self.cache_dir)

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path, metadata_path):
        output_text, metadata = "", []
        sentence_index = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                elements = [(float(w['top']), w['text']) for w in page.extract_words()]
                elements += [(float(img['top']), '[IMAGE]') for img in page.images]
                elements.sort(key=lambda x: x[0])
                page_text = " ".join(content for _, content in elements)
                metadata.append({"page_number": page_num, "start_sentence_index": sentence_index})
                output_text += page_text + " "
                sentence_index += sum(page_text.count(p) for p in ".!?")
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        return output_text.strip()

class TextProcessor:
    @staticmethod
    def calculate_token_histogram(tokenizer, text):
        sentences = text.split('. ')
        token_counts = defaultdict(int)
        for sentence in sentences:
            length = len(tokenizer(sentence, return_tensors="pt").input_ids[0])
            token_counts[length] += 1
        plt.bar(token_counts.keys(), token_counts.values(), color='skyblue')
        plt.xlabel('Token Count per Sentence')
        plt.ylabel('Frequency')
        plt.title('Token Count Histogram')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        return dict(token_counts)

class Translator:
    def __init__(self, model_handler):
        self.model = model_handler.model
        self.tokenizer = model_handler.tokenizer

    def split_text_into_chunks(self, text, max_tokens=250):
        sentences = text.split('. ')
        chunks, current_chunk = [], ""
        for sentence in sentences:
            temp_chunk = current_chunk + sentence + ". "
            if len(self.tokenizer(temp_chunk, return_tensors="pt").input_ids[0]) <= max_tokens:
                current_chunk = temp_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def translate(self, chunks, source_lang="Polish", target_lang="English"):
        output = []
        for chunk in tqdm(chunks, desc="Generating Translation"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Translate from {source_lang} to {target_lang}:\n{chunk}\n{target_lang}:"}],
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    num_beams=3,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )
                result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                match = re.search(r'\\?\[/INST\\?\](.*)', result, re.DOTALL)
                translated_text = match.group(1).strip() if match else result.strip()
                output.append(translated_text)
        return " ".join(output)

# Usage:
#model_handler = ModelHandler()

#metadata_path = pathlib.Path.cwd() / "Test_docs/metadata_pdf.json"
#pdf_path = pathlib.Path.cwd() / "Test_docs/Test_ALMA.pdf"

#pdf_text = PDFProcessor.extract_text(pdf_path, metadata_path)
#chunks = Translator(model_handler).split_text_into_chunks(pdf_text)
#translation = Translator(model_handler).translate(chunks)
#print(translation)
