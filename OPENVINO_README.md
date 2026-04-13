# OpenVINO 推理支持

本项目现在支持使用 **OpenVINO** 进行推理加速，可以在 CPU 上获得更好的性能表现。

## 安装 OpenVINO

首先需要安装 OpenVINO 及相关依赖：

```bash
pip install openvino>=2024.0.0 openvino-tokenizers>=2024.0.0
```

或者更新 requirements.txt 中的依赖：

```bash
pip install -r requirements.txt
```

## 转换模型为 OpenVINO 格式

在使用 OpenVINO 推理之前，需要先将 PyTorch 模型转换为 OpenVINO IR 格式：

```bash
python convert_to_openvino.py \
    --base_model_path /path/to/genshin-models \
    --charactor_model_path /path/to/character-model \
    --output_dir ./openvino_models \
    --precision FP16
```

### 参数说明：

- `--base_model_path`: Spark-TTS 基础模型路径（包含 Spark-TTS-0.5B 目录）
- `--charactor_model_path`: （可选）角色微调模型路径
- `--output_dir`: OpenVINO 模型输出目录
- `--precision`: 模型精度，可选 `FP16` 或 `FP32`（推荐 FP16 以获得更好性能）
- `--components`: 要转换的组件，可选 `llm`, `wav2vec2`, `bicodec`（默认全部转换）

### 示例：

```bash
# 转换所有组件
python convert_to_openvino.py \
    --base_model_path ./genshin \
    --charactor_model_path ./genshin/furina \
    --output_dir ./openvino_models/furina \
    --precision FP16

# 只转换 LLM 和 BiCodec
python convert_to_openvino.py \
    --base_model_path ./genshin \
    --output_dir ./openvino_models \
    --components llm bicodec
```

## 使用 OpenVINO 进行推理

转换完成后，使用 `openvino_infer.py` 进行推理：

```bash
python openvino_infer.py \
    --model_dir ./openvino_models/furina \
    --text "你好吗，今天过的怎么样呢？" \
    --output sparktts_furina.wav \
    --device CPU
```

### 参数说明：

- `--model_dir`: OpenVINO 模型目录（convert_to_openvino.py 的输出目录）
- `--text`: 要合成的文本
- `--output`: 输出音频文件名
- `--device`: 推理设备，可选 `CPU`, `GPU`, `AUTO`（默认 CPU）
- `--temperature`: 生成温度（默认 0.65）
- `--top_k`: Top-K 采样参数（默认 50）
- `--top_p`: Top-P 采样参数（默认 1.0）
- `--max_length`: 最大序列长度（默认 1024）

### 环境变量方式：

也可以使用环境变量设置参数：

```bash
export OV_MODEL_DIR=./openvino_models/furina
export OV_TEXT="你是说楼下罗森超市有活动吗？你这个老登怎么不早说！"
export OV_OUTPUT=furina_output.wav
export OV_DEVICE=CPU

python openvino_infer.py \
    --model_dir $OV_MODEL_DIR \
    --text "$OV_TEXT" \
    --output $OV_OUTPUT \
    --device $OV_DEVICE
```

## 性能对比

OpenVINO 优化后的模型在 Intel CPU 上通常可以获得：

- **2-5 倍** 的推理速度提升（相比 PyTorch CPU）
- **更低的内存占用**
- **更好的能效比**

实际性能取决于：
- CPU 型号（支持 AVX2/AVX-512/VNNI 指令集的 CPU 性能更好）
- 模型精度（FP16 比 FP32 更快）
- 输入文本长度

## 注意事项

1. **首次转换时间较长**：模型转换可能需要几分钟到十几分钟，取决于模型大小和硬件配置

2. **LLM 自回归生成**：当前的 OpenVINO 实现使用了简化的生成策略。对于完整的自回归生成（带 KV cache），建议使用 OpenVINO 的 Generate API 或参考官方示例

3. **BiCodec 组件**：BiCodec 被拆分为多个子组件（encoder, quantizer, speaker_encoder, prenet, postnet, decoder）分别转换和优化

4. **精度选择**：
   - `FP16`: 推荐用于 Intel CPU，速度快，质量损失极小
   - `FP32`: 如果需要最高质量，可以选择 FP32

5. **设备支持**：
   - `CPU`: 最稳定，适用于所有 Intel/AMD CPU
   - `GPU`: 需要 Intel 集成显卡或独立显卡
   - `AUTO`: 自动选择最佳设备

## 故障排除

### 问题：找不到 openvino_model.xml 文件

确保已经成功运行了 `convert_to_openvino.py`，并且指定的 `--model_dir` 是正确的输出目录。

### 问题：内存不足

尝试：
- 使用 `--precision FP16` 减少内存占用
- 减少 `--max_length` 参数
- 关闭其他占用内存的程序

### 问题：推理速度慢

检查：
- 是否使用了 `--device CPU`（而不是默认的 AUTO）
- CPU 是否支持 AVX2/AVX-512 指令集
- 系统是否有其他高负载进程

## 进阶使用

### 批量推理

可以修改 `openvino_infer.py` 添加批量处理功能：

```python
texts = ["文本 1", "文本 2", "文本 3"]
for i, text in enumerate(texts):
    inferencer.infer(text, output_filename=f"output_{i}.wav")
```

### 流式推理

对于长文本，可以实现流式生成，边生成边播放音频。

### 自定义采样策略

修改 `generate_tokens` 方法中的采样逻辑，实现更复杂的解码策略（如 beam search, diverse decoding 等）。

## 参考资料

- [OpenVINO 官方文档](https://docs.openvino.ai/)
- [OpenVINO 模型转换指南](https://docs.openvino.ai/latest/openvino_docs_model_convertor_MO_Overview.html)
- [Spark-TTS 原始项目](https://github.com/SparkAudio/Spark-TTS)
