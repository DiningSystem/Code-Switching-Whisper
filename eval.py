# scripts/eval.py
"""
Small inference demo to show how to force built-in Whisper task + language.
Usage:
    python .eval.py --model_dir ./outputs --audio examples/audio/file_0001.wav --task transcribe --language vietnamese
"""
import argparse
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_audio_as_input_features(processor, audio_path):
    speech_array, sr = sf.read(audio_path)
    if speech_array.ndim > 1:
        import numpy as np
        speech_array = np.mean(speech_array, axis=1)
    inputs = processor.feature_extractor(speech_array, sampling_rate=sr, return_tensors="pt")
    return inputs.input_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--language", choices=["english", "vietnamese"], default="vietnamese")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = load_audio_as_input_features(processor, args.audio).to(model.device)

    # get decoder prompt ids (this seeds startoftranscript + task + language tokens)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

    # generate (we do NOT suppress timestamps here)
    generated_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids, max_length=448, num_beams=5)
    text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Prediction:")
    print(text)

if __name__ == "__main__":
    main()
