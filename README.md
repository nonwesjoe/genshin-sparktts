# Genshin TTS
* the model files on ( https://huggingface.co/wesjos/spark-tts-genshin-charactors )
* each model named from charactors
* another important component 'audio tokenizer' file in 'Spark-TTS-0.5B'
# Usage
* python 3.12 suggested
* git clone this repo and cd into it.
* pip install -r requirements.txt
* change the model path and audio tokenizer path in the code.
* run the code by python infer.py
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