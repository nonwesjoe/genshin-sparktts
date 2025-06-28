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
input_text="你好吗,今天过的怎么样呢？"
model_path="/media/max/Hutao/genshin-tts-fp16/nilou"
base_model_path="/media/max/Hutao/genshin-tts-fp16/Spark-TTS-0.5B"
# You can also set the lora path if you want to use lora
if_lora=False
lora_path="/media/max/Hutao/genshin-tts-fp16/lora/furina"

##USE ENVIRONMENT VARIABLE TO SET THE MODEL PATH
if_lora=os.getenv('IF_LORA',if_lora)
input_text=os.getenv('INPUT_TEXT',input_text)
model_path=os.getenv('MODEL_PATH',model_path)
audio_tokenizer_path=os.getenv('BASE_MODEL_PATH',base_model_path)
base_model_path=os.path.join(os.getenv('BASE_MODEL_PATH',base_model_path),'LLM')
lora_path = os.getenv('LORA_PATH',lora_path)


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