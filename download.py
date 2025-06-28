from huggingface_hub import snapshot_download

charactor=input('which charactor? chose from [lora,furina,hutao,paimon,xiao,kazuha,nilou,zhongli,venti,...]')

based=snapshot_download(
    repo_id="wesjos/spark-tts-genshin-charactors",              # 仓库名
    local_dir="genshin",                   # 保存目录
    allow_patterns=["Spark-TTS-0.5B/*"],           # 只下载这个文件
)

saved=snapshot_download(
    repo_id="wesjos/spark-tts-genshin-charactors",              # 仓库名
    local_dir="genshin",                   # 保存目录
    allow_patterns=[f"{charactor}/*"],           # 只下载这个文件
)

print(f'base model is saved in {based}')
print(f'charactor model is saved in {saved}')