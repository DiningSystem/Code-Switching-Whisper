# data/resample_all.py
import os
import soundfile as sf
import numpy as np

# choose method: use librosa or torchaudio
USE_TORCHAUDIO = True

def resample_file(in_path, out_path, target_sr=16000):
    data, sr = sf.read(in_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr == target_sr:
        sf.write(out_path, data, target_sr)
        return
    if USE_TORCHAUDIO:
        try:
            import torch
            import torchaudio
            wav = torch.from_numpy(data.astype(np.float32))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            resampled = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
            if resampled.shape[0] == 1:
                res = resampled.squeeze(0).numpy()
            else:
                res = resampled.mean(dim=0).numpy()
            sf.write(out_path, res, target_sr)
            return
        except Exception:
            pass
    # fallback to librosa
    try:
        import librosa
        res = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sf.write(out_path, res, target_sr)
        return
    except Exception as e:
        raise RuntimeError("Install torchaudio or librosa to resample") from e

def resample_dir(input_dir, output_dir, target_sr=16000, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".wav", ".flac", ".mp3")):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.split('_')[0] + '.wav')  # keep same name
        if os.path.exists(out_path) and not overwrite:
            print("Skipping (exists):", out_path)
            continue
        print("Resampling:", in_path, "->", out_path)
        resample_file(in_path, out_path, target_sr=target_sr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    resample_dir(args.in_dir, args.out_dir, args.sr, args.overwrite)
