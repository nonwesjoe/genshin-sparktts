# Genshin TTS
* the model files on ( https://huggingface.co/wesjos/spark-tts-genshin-charactors )
* run this code online ( https://www.kaggle.com/code/suziwsz/genshin-sparktts )
* each model named from charactors
* another important component 'audio tokenizer' file in 'Spark-TTS-0.5B'
* available charactors: paimon, hutao, furina, kazuha, xiao, mona, ganyu, xiangling, shotgun, citlali, barbara, zhongli, venti, nahida, kaeya, yaoyao, yoimiya, nilou.(each charactor in one full finetuned model)
# Usage
* python 3.12 suggested
* <code>git clone https://github.com/nonwesjoe/genshin-sparktts.git && cd genshin-sparktts</code>
* when cuda is availabel, install torch 2.7.1 on cuda  <br><code>pip install torch torchaudio torchvision -i https://download.pytorch.org/whl/cu118/</code>  
else, install torch 2.7.1 on cpu  <br><code>pip install torch torchaudio torchvision -i https://download.pytorch.org/whl/cpu</code>
* install other requirements  <br><code>pip install -r requirements.txt</code>
* in terminal set some environment variables  <pre>
export CHARACTOR=nahida                                         # or other charactors  
export MODEL_PATH=/kaggle/working/genshin/                      # your model path  
export INPUT_TEXT="你是说楼下罗森超市有活动吗？你这个老登怎么不早说！"  # text to be converted  
##Lora adapater take smaller storage but not as good as full finetuned version  
export IF_LORA=''                                               # defalutly lora is not used  
export LORA_PATH=''                                             # lora path
</pre>

* download model files: defaultly, download one specific charactor model set in environment variable CHARACTOR. model will be download in ./genshin  <code>
python3 download.py
</code>

* run code to convert text to audio. audio outputs sparktts.wav.  <code>
python3 run.py
</code>

# Example
## poetry:
* kazuha(万叶):['play'](./examples/kazuha.wav)
* paimon(派蒙):['play'](./examples/paimon.wav)
* citlali(茜特拉莉)['play'](./examples/citlali.wav)
## congradulations:
* xiao(魈):['play'](./examples/xiao.wav)
* furina(芙宁娜):['play'](./examples/furina.wav)
* hutao(胡桃):['play'](./examples/hutao.wav)
## Acknowledgement
* [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
* [Genshin dataset](https://huggingface.co/datasets/simon3000/genshin-voice)
* [Unsloth](https://github.com/unslothai/unsloth)