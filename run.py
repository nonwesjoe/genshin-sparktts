from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from infer import infer
from peft import PeftModel
import os
device="cuda" if torch.cuda.is_available() else "cpu"
##SET ENVIRONMENT VARIABLE TO SET THE MODEL PATH
# os.environ["IF_LORA"] = ''
# os.environ["MODEL_PATH"] = "/media/max/Hutao/genshin-tts-fp16/furina"
# os.environ["BASE_MODEL_PATH"] = "/media/max/Hutao/genshin-tts-fp16/Spark-TTS-0.5B"
# os.environ["LORA_PATH"] = "/media/max/Hutao/genshin-tts-fp16/lora/hutao"
# os.environ["INPUT_TEXT"] = "我的天啊！今天超市大促销诶！我买了两瓶可乐，居然送了一包薯片呢。"

##USE ENVIRONMENT VARIABLE TO SET THE MODEL PATH
if_lora=os.getenv('IF_LORA',False)
input_text=os.getenv('INPUT_TEXT','你好吗,今天过的怎么样呢？')
model_path=os.getenv('MODEL_PATH','/media/max/Hutao/genshin-tts-fp16/nilou')
audio_tokenizer_path=os.getenv('BASE_MODEL_PATH','/media/max/Hutao/genshin-tts-fp16/Spark-TTS-0.5B')
base_model_path=os.path.join(os.getenv('BASE_MODEL_PATH','/media/max/Hutao/genshin-tts-fp16/Spark-TTS-0.5B'),'LLM')
lora_path = os.getenv('LORA_PATH','/media/max/Hutao/genshin-tts-fp16/lora/furina')

#max length of the audio part
max_seq_length=1024

if if_lora:
    print("using lora")
    model = AutoModelForCausalLM.from_pretrained(base_model_path,device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path,device)
else:
    print("using base model")
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path,device)


infer(model,tokenizer,audio_tokenizer,input_text,max_seq_length=max_seq_length,device=device)