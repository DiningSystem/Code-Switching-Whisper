# runner.py
import os
import random
import json
import yaml
import numpy as np
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor, WhisperForConditionalGeneration
from preprocess import prepare_example_map
from collate import DataCollatorSpeechSeq2Seq
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json_dataset(audio_dir: str, transcripts_json: str):
    with open(transcripts_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    if isinstance(data, dict):
        for filename, info in data.items():
            audio_path = os.path.join(audio_dir, filename)
            samples.append({"audio": audio_path, **info})
    elif isinstance(data, list):
        for info in data:
            audio_field = info.get("audio")
            if audio_field and not os.path.isabs(audio_field) and not os.path.exists(audio_field):
                audio_field = os.path.join(audio_dir, audio_field)
            samples.append({"audio": audio_field, **{k: v for k, v in info.items() if k != "audio"}})
    else:
        raise ValueError("transcripts.json must be list or dict")
    return Dataset.from_list(samples)

def run_train(config_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    model_id = cfg["model"]["model_id"]
    print("Loading processor and model:", model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    #model.config.use_cache = False
    #model.gradient_checkpointing_enable()

    ds = load_json_dataset(cfg["dataset"]["audio_dir"], cfg["dataset"]["transcripts_json"])
    ds = ds.train_test_split(test_size=cfg["dataset"].get("test_size", 0.1))

    task_probs = cfg["multitask"]["task_probabilities"]
    ds_pre = ds.map(lambda ex: prepare_example_map(ex, processor, task_probs),
                    remove_columns=ds["train"].column_names, num_proc=1)

    collator = DataCollatorSpeechSeq2Seq(processor=processor)

    tcfg = cfg["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=tcfg.get("output_dir", "./outputs"),
        overwrite_output_dir=True,
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 8),
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        gradient_accumulation_steps=4,
        learning_rate=float(tcfg.get("learning_rate", 3e-5)),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=tcfg.get("logging_steps", 50),
        eval_strategy="steps",
        save_strategy="epoch",
        label_smoothing_factor=0.0,
        eval_steps=1000,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=tcfg.get("fp16", False),
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_pre["train"],
        eval_dataset=ds_pre["test"],
        data_collator=collator,
        tokenizer=processor.tokenizer
    )

    print("Starting training ...")
    trainer.train()
    print("Saving model and processor to", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
