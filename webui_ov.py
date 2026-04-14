import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
from huggingface_hub import snapshot_download, HfApi
import gc
import psutil
import time
import threading

# Global state for loaded model
current_model = None
current_tokenizer = None
current_audio_tokenizer = None
current_character = None
model_lock = threading.Lock() # Lock to ensure thread safety for multi-user concurrent access

# Global state for shared UI status
global_model_status = "就绪 (Ready)"
global_infer_status = "就绪 (Ready)"

device = "cpu" # Force CPU for OpenVINO version
max_seq_length = 1024

def get_available_characters():
    if not os.path.exists("genshin_ov"):
        return []
    chars = [d for d in os.listdir("genshin_ov") if os.path.isdir(os.path.join("genshin_ov", d)) and d != "Spark-TTS-0.5B" and not d.startswith('.')]
    return sorted(chars)

def get_system_status():
    global current_character, global_model_status, global_infer_status
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        gpu_info = f" | GPU VRAM Allocated: {gpu_mem:.1f}GB"
    
    loaded_msg = f"当前加载的模型 (Loaded): {current_character if current_character else '无 (None)'}"
    sys_msg = f"CPU: {cpu}% | RAM: {mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB){gpu_info}"
    return loaded_msg, sys_msg, global_model_status, global_infer_status

def load_model(character, threads, enable_ht):
    global current_model, current_tokenizer, current_audio_tokenizer, current_character, global_model_status
    
    if not character:
        global_model_status = "Please select or enter a character name."
        yield global_model_status
        return
        
    if current_character == character:
        global_model_status = f"Model for {character} is already loaded."
        yield global_model_status
        return
        
    model_dir = os.path.join("genshin_ov", character)
    audio_tokenizer_dir = os.path.join("genshin_ov", "Spark-TTS-0.5B")
    
    if not os.path.exists(model_dir):
        global_model_status = f"Model for {character} not found locally in genshin_ov. Please export it first."
        yield global_model_status
        return
    if not os.path.exists(audio_tokenizer_dir):
        global_model_status = "Base model (Spark-TTS-0.5B) not found in genshin_ov. Please copy it or run export first."
        yield global_model_status
        return
        
    with model_lock:
        try:
            # Unload previous first if exists to save VRAM/RAM
            if current_model is not None:
                global_model_status = "Unloading previous model..."
                yield global_model_status
                unload_model_internal()
                
            global_model_status = f"Loading OpenVINO model for {character} with {threads} threads (HT: {enable_ht})..."
            yield global_model_status
            
            print(global_model_status)
            ov_config = {
                "INFERENCE_NUM_THREADS": int(threads),
                "ENABLE_HYPER_THREADING": "YES" if enable_ht else "NO"
            }
            current_model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU", ov_config=ov_config)
            current_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            current_audio_tokenizer = BiCodecTokenizer(audio_tokenizer_dir, device)
            current_character = character
            global_model_status = f"Successfully loaded OpenVINO model for {character} (Threads: {threads}, HT: {enable_ht})."
            yield global_model_status
        except Exception as e:
            global_model_status = f"Error loading model: {str(e)}"
            yield global_model_status

def unload_model_internal():
    global current_model, current_tokenizer, current_audio_tokenizer, current_character
    if current_model is not None:
        del current_model
        del current_tokenizer
        del current_audio_tokenizer
        current_model = None
        current_tokenizer = None
        current_audio_tokenizer = None
        current_character = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Model unloaded successfully."
    return "No model is currently loaded."

def unload_model():
    global global_model_status
    global_model_status = "Unloading model..."
    yield global_model_status
    with model_lock:
        global_model_status = unload_model_internal()
        yield global_model_status

def generate_audio(text):
    global current_model, current_tokenizer, current_audio_tokenizer, current_character, global_infer_status
    
    if current_model is None:
        global_infer_status = "❌ 错误：请先加载一个角色模型。"
        yield None, global_infer_status
        return
        
    if not text:
        global_infer_status = "❌ 错误：请输入需要合成的文本。"
        yield None, global_infer_status
        return
        
    global_infer_status = f"正在生成音频 (Generating): {text[:20]}..."
    yield None, global_infer_status
        
    with model_lock:
        try:
            from infer import generate_speech_from_text
            
            start_time = time.time()
            
            generated_waveform = generate_speech_from_text(
                text, 
                current_model, 
                current_tokenizer, 
                current_audio_tokenizer,
                max_seq_length=max_seq_length,
                device="cpu"
            )
            
            elapsed_time = time.time() - start_time
            
            if generated_waveform.size > 0:
                output_filename = "temp_output.wav"
                sample_rate = current_audio_tokenizer.config.get("sample_rate", 16000)
                sf.write(output_filename, generated_waveform, sample_rate)
                global_infer_status = f"✅ 生成成功！\n耗时: {elapsed_time:.2f} 秒\n文本: {text}"
                yield output_filename, global_infer_status
            else:
                global_infer_status = "❌ 错误：音频生成失败（未找到有效的音频 token）。"
                yield None, global_infer_status
        except Exception as e:
            global_infer_status = f"❌ 错误：生成过程中出现异常: {str(e)}"
            yield None, global_infer_status

def update_choices():
    return gr.update(choices=get_available_characters())

with gr.Blocks(title="Genshin Spark-TTS OpenVINO WebUI") as app:
    gr.Markdown("# Genshin Spark-TTS OpenVINO WebUI\n可以选择角色加载 OpenVINO 模型，体验 CPU 上的加速推理。")
    
    with gr.Row():
        current_model_info = gr.Textbox(label="当前加载模型 (Current Model)", interactive=False)
        system_stats_info = gr.Textbox(label="系统资源 (System Stats)", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 模型管理 (Model Management)")
            char_dropdown = gr.Dropdown(choices=get_available_characters(), label="选择角色 (Select Character)", info="只能选择已经导出到 genshin_ov 的角色")
            refresh_btn = gr.Button("刷新角色列表 (Refresh List)")
            
            with gr.Row():
                threads_slider = gr.Slider(minimum=1, maximum=os.cpu_count() or 16, value=(os.cpu_count() or 8), step=1, label="推理线程数 (Inference Threads)")
                ht_checkbox = gr.Checkbox(label="启用超线程 (Enable Hyper-Threading)", value=True, info="如果部署在服务器上，建议关闭此项只使用物理核。")
            
            with gr.Row():
                load_btn = gr.Button("加载模型 (Load)", variant="primary")
                unload_btn = gr.Button("卸载模型 (Unload)")
                
            status_text = gr.Textbox(label="状态信息 (Status)", interactive=False)
            
        with gr.Column():
            gr.Markdown("### 2. 推理 (Inference)")
            input_text = gr.Textbox(label="输入文本 (Input Text)", lines=3, placeholder="输入需要合成的文本...")
            generate_btn = gr.Button("生成音频 (Generate Audio)", variant="primary")
            output_audio = gr.Audio(label="生成的音频 (Generated Audio)", type="filepath")
            infer_status = gr.Textbox(label="推理状态 (Inference Status)", interactive=False)

    # Event handlers
    refresh_btn.click(fn=update_choices, outputs=char_dropdown)
    load_btn.click(fn=load_model, inputs=[char_dropdown, threads_slider, ht_checkbox], outputs=[status_text])
    unload_btn.click(fn=unload_model, outputs=[status_text])
    
    generate_btn.click(fn=generate_audio, inputs=[input_text], outputs=[output_audio, infer_status])
    
    gr.Timer(2).tick(fn=get_system_status, inputs=None, outputs=[current_model_info, system_stats_info, status_text, infer_status])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7862, share=False)
