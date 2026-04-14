import os
import shutil
import argparse
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Export Genshin Spark-TTS models to OpenVINO FP16 format.")
    parser.add_argument("--src_dir", type=str, default="genshin", help="Source directory containing the models")
    parser.add_argument("--dst_dir", type=str, default="genshin_ov", help="Destination directory for OpenVINO models")
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir

    os.makedirs(dst_dir, exist_ok=True)

    # 1. Copy the base tokenizer/vocoder model as it is.
    # The BiCodecTokenizer is a custom PyTorch module, and we'll keep it running in PyTorch
    # since it's a small part of the pipeline compared to the LLM.
    src_base = os.path.join(src_dir, "Spark-TTS-0.5B")
    dst_base = os.path.join(dst_dir, "Spark-TTS-0.5B")
    if os.path.exists(src_base):
        if not os.path.exists(dst_base):
            print(f"Copying base model {src_base} to {dst_base}...")
            shutil.copytree(src_base, dst_base)
        else:
            print(f"Base model already exists at {dst_base}.")
    else:
        print(f"Warning: Base model not found at {src_base}. Please download it first.")

    # 2. Find all character models
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    chars = [d for d in os.listdir(src_dir) 
             if os.path.isdir(os.path.join(src_dir, d)) 
             and d != "Spark-TTS-0.5B" 
             and not d.startswith('.')]

    print(f"Found characters to export: {chars}")

    # 3. Export each character model
    for char in chars:
        src_model_path = os.path.join(src_dir, char)
        dst_model_path = os.path.join(dst_dir, char)
        
        if os.path.exists(os.path.join(dst_model_path, "openvino_model.xml")):
            print(f"Skipping {char}, already exported at {dst_model_path}")
            continue
            
        print(f"\n==================================================")
        print(f"Exporting {char} to OpenVINO FP16 format...")
        print(f"==================================================")
        
        try:
            # We use exportConfig to specify we want fp16 weights
            # stateful=True is highly recommended for CausalLM models (KV Cache)
            ov_model = OVModelForCausalLM.from_pretrained(
                src_model_path, 
                export=True, 
                compile=False,
                stateful=True,
            )
            
            # Convert the model to fp16
            ov_model.half()
            
            # Save the converted model
            ov_model.save_pretrained(dst_model_path)
            
            # Save the tokenizer as well
            tokenizer = AutoTokenizer.from_pretrained(src_model_path)
            tokenizer.save_pretrained(dst_model_path)
            
            print(f"✅ Successfully exported {char} to {dst_model_path}")
        except Exception as e:
            print(f"❌ Failed to export {char}: {e}")

if __name__ == "__main__":
    main()