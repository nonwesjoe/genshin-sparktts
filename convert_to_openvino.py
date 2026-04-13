"""
Convert Spark-TTS models to OpenVINO IR format for optimized CPU inference.

This script converts:
1. The LLM model (AutoModelForCausalLM) to OpenVINO
2. The BiCodec model components to OpenVINO
3. The Wav2Vec2 feature extractor to OpenVINO
"""

import os
import torch
import openvino as ov
from openvino.tools import mo
from pathlib import Path
import argparse

# Set environment variable for OpenVINO conversion
os.environ["OPENVINO_LOG_LEVEL"] = "INFO"


def convert_llm_to_openvino(model_path: str, output_dir: str, precision: str = "FP16"):
    """
    Convert the LLM model to OpenVINO IR format.
    
    Args:
        model_path: Path to the HuggingFace model directory
        output_dir: Output directory for OpenVINO models
        precision: Model precision (FP16 or FP32)
    """
    print(f"\n{'='*60}")
    print(f"Converting LLM model from {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Precision: {precision}")
    print(f"{'='*60}\n")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Convert to FP32 first for stability
        device_map="cpu"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create dummy input for tracing
    prompt = "<|task_tts|><|start_content|>test<|end_content|><|start_global_token|>"
    model_inputs = tokenizer([prompt], return_tensors="pt")
    
    # Define input shapes for dynamic axes
    batch_size = 1
    seq_length = model_inputs.input_ids.shape[1]
    max_seq_length = 1024
    
    # Convert using OpenVINO's converter
    print("Converting model to OpenVINO IR...")
    
    ov_model = ov.convert_model(
        model,
        example_input=(model_inputs.input_ids,),
        input=[ov.PartialShape([batch_size, ov.Dimension(-1, max_seq_length)])],
    )
    
    # Apply FP16 compression if requested
    if precision == "FP16":
        print("Applying FP16 compression...")
        ov.compress_weights(ov_model, mode=ov.CompressMode.FP16)
    
    # Save the model
    llm_output_dir = Path(output_dir) / "LLM"
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    ov.save_model(ov_model, str(llm_output_dir / "openvino_model.xml"))
    print(f"LLM model saved to {llm_output_dir}")
    
    # Save tokenizer config
    tokenizer.save_pretrained(str(llm_output_dir))
    print("Tokenizer saved")
    
    return str(llm_output_dir)


def convert_wav2vec2_to_openvino(model_path: str, output_dir: str, precision: str = "FP16"):
    """
    Convert Wav2Vec2 feature extractor to OpenVINO IR format.
    
    Args:
        model_path: Path to the wav2vec2 model directory
        output_dir: Output directory for OpenVINO models
        precision: Model precision
    """
    print(f"\n{'='*60}")
    print(f"Converting Wav2Vec2 model from {model_path}")
    print(f"{'='*60}\n")
    
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    
    # Load model and processor
    print("Loading Wav2Vec2 model...")
    model = Wav2Vec2Model.from_pretrained(model_path, torch_dtype=torch.float32)
    model.config.output_hidden_states = True
    model.eval()
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    # Create dummy input
    sample_rate = 16000
    duration = 1.0  # seconds
    dummy_audio = torch.randn(int(sample_rate * duration))
    
    inputs = processor(
        [dummy_audio.numpy()],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    
    # Convert to OpenVINO
    print("Converting to OpenVINO IR...")
    
    ov_model = ov.convert_model(
        model,
        example_input=(inputs.input_values,),
        input=[ov.PartialShape([1, ov.Dimension(-1)])],  # Dynamic sequence length
    )
    
    # Apply precision
    if precision == "FP16":
        print("Applying FP16 compression...")
        ov.compress_weights(ov_model, mode=ov.CompressMode.FP16)
    
    # Save model
    wav2vec_output_dir = Path(output_dir) / "wav2vec2-large-xlsr-53"
    wav2vec_output_dir.mkdir(parents=True, exist_ok=True)
    
    ov.save_model(ov_model, str(wav2vec_output_dir / "openvino_model.xml"))
    print(f"Wav2Vec2 model saved to {wav2vec_output_dir}")
    
    # Save processor config
    processor.save_pretrained(str(wav2vec_output_dir))
    print("Processor config saved")
    
    return str(wav2vec_output_dir)


def convert_bicodec_to_openvino(model_dir: str, output_dir: str, precision: str = "FP16"):
    """
    Convert BiCodec model to OpenVINO IR format.
    
    This converts the encoder, quantizer, speaker encoder, prenet, postnet,
    and wave generator components separately for better optimization.
    
    Args:
        model_dir: Path to BiCodec model directory
        output_dir: Output directory for OpenVINO models
        precision: Model precision
    """
    print(f"\n{'='*60}")
    print(f"Converting BiCodec model from {model_dir}")
    print(f"{'='*60}\n")
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from sparktts.models.bicodec import BiCodec
    from sparktts.utils.file import load_config
    
    # Load config and model
    print("Loading BiCodec model...")
    config = load_config(f'{model_dir}/config.yaml')['audio_tokenizer']
    
    model = BiCodec.load_from_checkpoint(model_dir)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    duration = 1.0
    feat_dim = 1024
    feat_length = int(duration * 50)  # 50 Hz frame rate
    mel_dim = config["num_mels"]
    mel_length = int(duration * 16000 / config["hop_length"])
    
    # Random test inputs
    feat = torch.randn(batch_size, feat_length, feat_dim)
    mel = torch.randn(batch_size, mel_length, mel_dim)
    ref_wav = torch.randn(batch_size, int(duration * 16000))
    
    batch = {
        "feat": feat,
        "ref_wav": ref_wav,
    }
    
    # Convert each component separately for better optimization
    
    # 1. Feature Encoder
    print("\nConverting Feature Encoder...")
    with torch.no_grad():
        encoder_ov = ov.convert_model(
            model.encoder,
            example_input=(feat.transpose(1, 2),),
        )
        
        if precision == "FP16":
            ov.compress_weights(encoder_ov, mode=ov.CompressMode.FP16)
        
        encoder_dir = Path(output_dir) / "BiCodec" / "encoder"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(encoder_ov, str(encoder_dir / "openvino_model.xml"))
        print(f"Encoder saved to {encoder_dir}")
    
    # 2. Quantizer
    print("\nConverting Quantizer...")
    with torch.no_grad():
        z = model.encoder(feat.transpose(1, 2))
        quantizer_ov = ov.convert_model(
            model.quantizer,
            example_input=(z,),
        )
        
        if precision == "FP16":
            ov.compress_weights(quantizer_ov, mode=ov.CompressMode.FP16)
        
        quantizer_dir = Path(output_dir) / "BiCodec" / "quantizer"
        quantizer_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantizer_ov, str(quantizer_dir / "openvino_model.xml"))
        print(f"Quantizer saved to {quantizer_dir}")
    
    # 3. Speaker Encoder
    print("\nConverting Speaker Encoder...")
    with torch.no_grad():
        mel_spec = model.mel_transformer(ref_wav).squeeze(1)
        speaker_enc_ov = ov.convert_model(
            model.speaker_encoder,
            example_input=(mel_spec.transpose(1, 2),),
        )
        
        if precision == "FP16":
            ov.compress_weights(speaker_enc_ov, mode=ov.CompressMode.FP16)
        
        speaker_dir = Path(output_dir) / "BiCodec" / "speaker_encoder"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(speaker_enc_ov, str(speaker_dir / "openvino_model.xml"))
        print(f"Speaker encoder saved to {speaker_dir}")
    
    # 4. Prenet
    print("\nConverting Prenet...")
    with torch.no_grad():
        z_q = model.quantizer.z_q if hasattr(model.quantizer, 'z_q') else model.quantizer(z)[0]
        d_vector = model.speaker_encoder(mel_spec.transpose(1, 2))[1]
        prenet_ov = ov.convert_model(
            model.prenet,
            example_input=(z_q, d_vector),
        )
        
        if precision == "FP16":
            ov.compress_weights(prenet_ov, mode=ov.CompressMode.FP16)
        
        prenet_dir = Path(output_dir) / "BiCodec" / "prenet"
        prenet_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(prenet_ov, str(prenet_dir / "openvino_model.xml"))
        print(f"Prenet saved to {prenet_dir}")
    
    # 5. Postnet
    print("\nConverting Postnet...")
    with torch.no_grad():
        x = model.prenet(z_q, d_vector)
        postnet_ov = ov.convert_model(
            model.postnet,
            example_input=(x,),
        )
        
        if precision == "FP16":
            ov.compress_weights(postnet_ov, mode=ov.CompressMode.FP16)
        
        postnet_dir = Path(output_dir) / "BiCodec" / "postnet"
        postnet_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(postnet_ov, str(postnet_dir / "openvino_model.xml"))
        print(f"Postnet saved to {postnet_dir}")
    
    # 6. Wave Generator (Decoder)
    print("\nConverting Wave Generator...")
    with torch.no_grad():
        x = model.prenet(z_q, d_vector)
        x_with_cond = x + d_vector.unsqueeze(-1)
        decoder_ov = ov.convert_model(
            model.decoder,
            example_input=(x_with_cond,),
        )
        
        if precision == "FP16":
            ov.compress_weights(decoder_ov, mode=ov.CompressMode.FP16)
        
        decoder_dir = Path(output_dir) / "BiCodec" / "decoder"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(decoder_ov, str(decoder_dir / "openvino_model.xml"))
        print(f"Wave generator saved to {decoder_dir}")
    
    # Save config file
    import shutil
    bicodec_output_dir = Path(output_dir) / "BiCodec"
    bicodec_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{model_dir}/config.yaml', str(bicodec_output_dir / 'config.yaml'))
    print("Config file copied")
    
    return str(bicodec_output_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert Spark-TTS models to OpenVINO")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base Spark-TTS model directory")
    parser.add_argument("--charactor_model_path", type=str, default=None,
                        help="Path to character-specific fine-tuned model (optional)")
    parser.add_argument("--output_dir", type=str, default="./openvino_models",
                        help="Output directory for OpenVINO models")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP16", "FP32"],
                        help="Model precision (default: FP16)")
    parser.add_argument("--components", type=str, nargs="+", 
                        default=["llm", "wav2vec2", "bicodec"],
                        choices=["llm", "wav2vec2", "bicodec"],
                        help="Components to convert (default: all)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Spark-TTS to OpenVINO Converter")
    print("="*60)
    print(f"Base model path: {args.base_model_path}")
    if args.charactor_model_path:
        print(f"Character model path: {args.charactor_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Precision: {args.precision}")
    print(f"Components: {args.components}")
    print("="*60)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert components
    if "llm" in args.components:
        llm_path = os.path.join(args.base_model_path, "Spark-TTS-0.5B/LLM")
        if args.charactor_model_path:
            llm_path = os.path.join(args.charactor_model_path, "LLM")
        convert_llm_to_openvino(llm_path, args.output_dir, args.precision)
    
    if "wav2vec2" in args.components:
        wav2vec_path = os.path.join(args.base_model_path, "Spark-TTS-0.5B/wav2vec2-large-xlsr-53")
        convert_wav2vec2_to_openvino(wav2vec_path, args.output_dir, args.precision)
    
    if "bicodec" in args.components:
        bicodec_path = os.path.join(args.base_model_path, "Spark-TTS-0.5B/BiCodec")
        convert_bicodec_to_openvino(bicodec_path, args.output_dir, args.precision)
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print(f"OpenVINO models saved to: {args.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Install openvino package: pip install openvino openvino-tokenizers")
    print("2. Run inference with: python openvino_infer.py --model_dir <output_dir>")


if __name__ == "__main__":
    main()
