import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
from huggingface_hub import snapshot_download
import gc

# Global state for loaded model
current_model = None
current_tokenizer = None
current_audio_tokenizer = None
current_character = None

device = "cuda" if torch.cuda.is_available() else "cpu"
max_seq_length = 1024

def get_available_characters():
    if not os.path.exists("genshin"):
        return []
    chars = [d for d in os.listdir("genshin") if os.path.isdir(os.path.join("genshin", d)) and d != "Spark-TTS-0.5B"]
    return sorted(chars)

def load_model(character):
    global current_model, current_tokenizer, current_audio_tokenizer, current_character
    
    if not character:
        return "Please select or enter a character name."
        
    model_dir = os.path.join("genshin", character)
    audio_tokenizer_dir = os.path.join("genshin", "Spark-TTS-0.5B")
    
    if not os.path.exists(model_dir):
        return f"Model for {character} not found locally. Please download it first."
    if not os.path.exists(audio_tokenizer_dir):
        return "Base model (Spark-TTS-0.5B) not found. Please download it first."
        
    try:
        # Unload previous first if exists to save VRAM
        if current_model is not None:
            unload_model()
            
        print(f"Loading model for {character}...")
        current_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device)
        current_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        current_audio_tokenizer = BiCodecTokenizer(audio_tokenizer_dir, device)
        current_character = character
        return f"Successfully loaded model for {character}."
    except Exception as e:
        return f"Error loading model: {str(e)}"

def unload_model():
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

def download_character(character):
    if not character:
        return "Please enter a character name to download.", gr.update()
    
    try:
        # Download base model if not exists
        if not os.path.exists(os.path.join("genshin", "Spark-TTS-0.5B")):
            snapshot_download(
                repo_id="wesjos/spark-tts-genshin-charactors",
                local_dir="genshin",
                allow_patterns=["Spark-TTS-0.5B/*"],
            )
            
        # Download character model
        snapshot_download(
            repo_id="wesjos/spark-tts-genshin-charactors",
            local_dir="genshin",
            allow_patterns=[f"{character}/*"],
        )
        chars = get_available_characters()
        return f"Successfully downloaded {character}.", gr.update(choices=chars, value=character)
    except Exception as e:
        return f"Error downloading {character}: {str(e)}", gr.update()

def generate_audio(text):
    global current_model, current_tokenizer, current_audio_tokenizer, current_character
    
    if current_model is None:
        return None, "Please load a model first."
        
    if not text:
        return None, "Please enter some text."
        
    try:
        from infer import generate_speech_from_text
        generated_waveform = generate_speech_from_text(
            text, 
            current_model, 
            current_tokenizer, 
            current_audio_tokenizer,
            max_seq_length=max_seq_length,
            device=device
        )
        
        if generated_waveform.size > 0:
            output_filename = "temp_output.wav"
            sample_rate = current_audio_tokenizer.config.get("sample_rate", 16000)
            sf.write(output_filename, generated_waveform, sample_rate)
            return output_filename, "Generation successful!"
        else:
            return None, "Audio generation failed (no tokens found)."
    except Exception as e:
        return None, f"Error during generation: {str(e)}"

def update_choices():
    return gr.update(choices=get_available_characters())

with gr.Blocks(title="Genshin Spark-TTS WebUI") as app:
    gr.Markdown("# Genshin Spark-TTS WebUI\n可以选择角色加载模型，也可以卸载模型以释放显存。")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 模型管理 (Model Management)")
            char_dropdown = gr.Dropdown(choices=get_available_characters(), label="选择角色 (Select Character)", allow_custom_value=True, info="可直接输入未下载的角色名并点击下载")
            refresh_btn = gr.Button("刷新角色列表 (Refresh List)")
            
            with gr.Row():
                download_btn = gr.Button("下载角色模型 (Download)")
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
    download_btn.click(fn=download_character, inputs=[char_dropdown], outputs=[status_text, char_dropdown])
    load_btn.click(fn=load_model, inputs=[char_dropdown], outputs=[status_text])
    unload_btn.click(fn=unload_model, outputs=[status_text])
    
    generate_btn.click(fn=generate_audio, inputs=[input_text], outputs=[output_audio, infer_status])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
