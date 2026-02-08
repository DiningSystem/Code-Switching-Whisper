# preprocess.py
import random
import soundfile as sf
import numpy as np

START_TOKEN = "<|START OF TRANSCRIPT|>"
END_TOKEN = "<|EOT|>"

def convert_inline_tags_to_whisper(text: str) -> str:
    """
    Convert inline [vi]/[en] -> Whisper tokens <|vi|> / <|en|>.
    Safe if text is None.
    """
    if text is None:
        return ""
    # Simple replace; extend if dataset uses other bracket tags.
    return text.replace("[vi]", "<|vi|>").replace("[en]", "<|en|>")

def build_target_whisper_format(item: dict, task_probs: dict):
    """
    Choose a task according to task_probs and build the label string using Whisper built-in tokens.
    Returns (label_text, chosen_task)
    """
    tasks = list(task_probs.keys())
    weights = list(task_probs.values())
    choice = random.choices(tasks, weights=weights, k=1)[0]

    if choice == "transcribe":
        # Keep inline tags converted; do not re-add a separate language token to avoid duplication.
        code_switch = convert_inline_tags_to_whisper(item.get("code_switch", ""))
        # Structure: <|startoftranscript|><|transcribe|> CODE_SWITCH <|endoftext|>
        label = f"{START_TOKEN} <|TRANSCRIBE|> {code_switch} {END_TOKEN}"
    elif choice == "translate_vi":
        vi_full = item.get("vi_full", "")
        # We seed with the target language token to be explicit
        label = f"{START_TOKEN} <|TRANSLATE|> <|NOTIMESTAMPS|> {vi_full} {END_TOKEN}"
    else:  # translate_en
        en_full = item.get("en_full", "")
        label = f"{START_TOKEN} <|TRANSLATE|> {en_full} {END_TOKEN}"

    return label, choice

def prepare_example_map(example: dict, processor, task_probs: dict):
    """
    Given an example dict with 'audio' path and text fields, return
    a dict containing numpy input_features and label input ids under 'labels'.
    """
    audio_path = example["audio"]
    speech_array, sr = sf.read(audio_path)
    # ensure mono float32
    if speech_array.ndim > 1:
        # convert to mono by averaging channels
        speech_array = np.mean(speech_array, axis=1)
    speech_array = speech_array.astype("float32")

    # extract input features (encoder inputs)
    input_features = processor.feature_extractor(
        speech_array, sampling_rate=sr, return_tensors="np"
    ).input_features[0]

    # build target label string and tokenize (Whisper tokenizer -> uses built-in tokens)
    target_text, chosen_task = build_target_whisper_format(example, task_probs)
    labels = processor.tokenizer(target_text, add_special_tokens=True, return_tensors="np").input_ids[0]

    return {"input_features": input_features, "labels": labels, "task": chosen_task}
