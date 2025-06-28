from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from infer import infer
from peft import PeftModel
import os
device="cuda" if torch.cuda.is_available() else "cpu"
#train on max seq length of 1024
max_seq_length=512

##Set variables
charactor='hutao'
input_text="你好吗,今天过的怎么样呢？"
model_path="/media/max/Hutao/genshin-tts-fp16/"
# You can also set the lora path if you want to use lora
if_lora=False
lora_path="genshin/lora"

##USE ENVIRONMENT VARIABLE TO SET THE MODEL PATH
if_lora=os.getenv('IF_LORA',if_lora)
charactor=os.getenv('CHARACTOR',charactor)
input_text=os.getenv('INPUT_TEXT',input_text)
model_path=os.path.join(os.getenv('MODEL_PATH',model_path),charactor)
audio_tokenizer_path=os.path.join(os.getenv('MODEL_PATH',model_path),'Spark-TTS-0.5B')
base_model_path=os.path.join(os.getenv('BASE_MODEL_PATH',model_path),'Spark-TTS-0.5B/LLM')
lora_path = os.path.join(os.getenv('LORA_PATH',lora_path),charactor)


if if_lora:
    print("Using lora adapter on base model")
    model = AutoModelForCausalLM.from_pretrained(base_model_path,device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path,device)
else:
    print("Using full finetuned model")
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path,device)


infer(model,tokenizer,audio_tokenizer,input_text,max_seq_length=max_seq_length,device=device)