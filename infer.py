import re
import torch
import numpy as np
import soundfile as sf
from playsound import playsound

@torch.inference_mode()
def generate_speech_from_text(
    text: str,
    model,
    tokenizer,
    audio_tokenizer,
    temperature=0.65,   # Generation temperature
    top_k= 50,            # Generation top_k
    top_p= 1,        # Generation top_p
    max_seq_length= 1024, # Max tokens for audio part
    device='cuda' if torch.cuda.is_available() else 'cpu'):

    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])

    # print(f'prompt: {prompt}')

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print("Generating token sequence...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_seq_length, # Limit generation length
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id, # Stop token
        pad_token_id=tokenizer.pad_token_id # Use models pad token id
    )
    print("Token sequence generated.")
    # print(f'length of generated_ids: {len(generated_ids[0])}')

    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]

    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging

    # Extract semantic token IDs using regex
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found in the generated output.")
        # Handle appropriately - perhaps return silence or raise error
        return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0) # Add batch dim

    # Extract global token IDs using regex (assuming controllable mode also generates these)
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    # print(global_matches)
    if not global_matches:
        print("Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail.")
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
    else:
        pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0) # Add batch dim
    


    pred_global_ids = pred_global_ids.unsqueeze(0) # Shape becomes (1, 1, N_global)

    # print(f'global_ids: {pred_global_ids}')
    # print(f'semantic_ids: {pred_semantic_ids}')

    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
    print(f"Found {pred_global_ids.shape[2]} global tokens.")


    # 5. Detokenize using BiCodecTokenizer
    print("Detokenizing audio tokens...")
    # Ensure audio_tokenizer and its internal model are on the correct device
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    # Squeeze the extra dimension from global tokens as seen in SparkTTS example
    
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device).squeeze(0), # Shape (1, N_global)
        pred_semantic_ids.to(device)           # Shape (1, N_semantic)
    )
    print("Detokenization complete.")
    # print(f'wav np shape:{wav_np.shape}')
    return wav_np

def infer(model, tokenizer, audio_tokenizer,input_text,**kwargs):
    print(f"Generating speech for: '{input_text}'")
    generated_waveform = generate_speech_from_text(input_text, model, tokenizer, audio_tokenizer,**kwargs)

    if generated_waveform.size > 0:
        output_filename = "sparktts.wav"
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_filename, generated_waveform, sample_rate)
        print(f"Audio saved to {output_filename}")
        playsound(output_filename)
    else:
        print("Audio generation failed (no tokens found?).")

