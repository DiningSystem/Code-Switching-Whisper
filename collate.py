# collate.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: object  # WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [f["input_features"] for f in features]
        batch_inputs = self.processor.feature_extractor.pad({"input_features": input_features}, return_tensors="pt")

        labels = [f["labels"] for f in features]
        max_len = max(len(l) for l in labels)
        padded = torch.full((len(labels), max_len), fill_value=self.processor.tokenizer.pad_token_id, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        padded[padded == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": batch_inputs["input_features"], "labels": padded}
