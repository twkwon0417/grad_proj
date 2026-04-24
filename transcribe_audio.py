"""
Faster-Whisper large-v3 (FP16) STT script for Unified KEMDy dataset.
Generates transcriptions mapped to metadata.csv for training use.

Usage:
    python transcribe_audio.py

Requirements:
    pip install faster-whisper pandas tqdm
"""
import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm


def transcribe_dataset(
    metadata_path: str,
    audio_dir: str,
    output_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "ko",
    beam_size: int = 5,
    model_size: str = "large-v3",
):
    # Load Faster-Whisper large-v3 in FP16
    print(f"[Model] Loading Whisper {model_size} on {device} ({compute_type})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    print("[Model] Loaded.")

    # Load metadata (source of truth for mapping)
    metadata_df = pd.read_csv(metadata_path)
    print(f"[Data] {len(metadata_df)} rows in metadata.")

    # Resume support: skip already processed rows
    processed = set()
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            done_df = pd.read_csv(output_path)
            processed = set(done_df["filename"].astype(str).tolist())
            print(f"[Resume] {len(processed)} files already transcribed. Skipping.")
        except Exception:
            file_exists = False

    # Columns preserve all metadata fields + STT output for easy training
    fieldnames = [
        "filename",
        "dataset_source",
        "Emotion",
        "Valence",
        "Arousal",
        "transcription",
        "language",
        "language_probability",
        "duration",
    ]

    mode = "a" if file_exists else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for _, row in tqdm(
            metadata_df.iterrows(),
            total=len(metadata_df),
            desc="Transcribing",
        ):
            filename = row["filename"]
            if filename in processed:
                continue

            audio_path = os.path.join(audio_dir, filename)
            if not os.path.exists(audio_path):
                continue

            try:
                segments, info = model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=True,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()

                writer.writerow({
                    "filename": filename,
                    "dataset_source": row["dataset_source"],
                    "Emotion": row["Emotion"],
                    "Valence": row["Valence"],
                    "Arousal": row["Arousal"],
                    "transcription": text,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 3),
                })
                f.flush()
            except Exception as e:
                # Write empty transcription with error marker so mapping is preserved
                writer.writerow({
                    "filename": filename,
                    "dataset_source": row["dataset_source"],
                    "Emotion": row["Emotion"],
                    "Valence": row["Valence"],
                    "Arousal": row["Arousal"],
                    "transcription": "",
                    "language": "",
                    "language_probability": 0.0,
                    "duration": 0.0,
                })
                f.flush()
                print(f"\n[Error] {filename}: {e}")

    print(f"\n[Done] Transcriptions saved: {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(base, "data", "processed", "Unified_KEMDy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=os.path.join(default_root, "metadata.csv"))
    parser.add_argument("--audio_dir", default=os.path.join(default_root, "Audio_Files"))
    parser.add_argument("--output", default=os.path.join(default_root, "transcriptions.csv"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--compute_type", default="float16",
                        choices=["float16", "float32", "int8", "int8_float16"])
    parser.add_argument("--language", default="ko")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--model_size", default="large-v3")
    args = parser.parse_args()

    transcribe_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_path=args.output,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        model_size=args.model_size,
    )
