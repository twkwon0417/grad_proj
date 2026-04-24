import os
import glob
import shutil
import pandas as pd

def process_datasets(base_path, target_dir):
    """
    Unifies KEMDy18, KEMDy19, and KEMDy20 datasets into a single directory.
    Filters by specific emotions and generates a consolidated metadata.csv.
    """
    target_emotions = {'neutral', 'sad', 'happy', 'angry'}
    
    output_audio_dir = os.path.join(target_dir, 'Audio_Files')
    os.makedirs(output_audio_dir, exist_ok=True)
    
    metadata_records = []
    
    # Collect all wav files and create a mapping of filename to its absolute path
    wav_path_map = {}
    for dataset_name in ['KESDy18', 'KEMDy19_v1_4', 'KEMDy20_v1_3']:
        dataset_dir = os.path.join(base_path, dataset_name)
        if not os.path.exists(dataset_dir):
            continue

        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_path_map[file] = os.path.join(root, file)

    def add_to_metadata(dataset_source, original_wav_name, emotion, valence, arousal):
        """
        Validates emotion and copies the file if it exists in the mapping.
        """
        if pd.isna(emotion):
            return
            
        emotion_cleaned = str(emotion).strip().lower()
        if emotion_cleaned not in target_emotions:
            return
            
        if original_wav_name not in wav_path_map:
            return
            
        source_wav_path = wav_path_map[original_wav_name]
        new_wav_name = f"{dataset_source}_{original_wav_name}"
        dest_wav_path = os.path.join(output_audio_dir, new_wav_name)
        
        shutil.copy2(source_wav_path, dest_wav_path)
        
        metadata_records.append({
            'filename': new_wav_name,
            'dataset_source': dataset_source,
            'Emotion': emotion_cleaned,
            'Valence': valence,
            'Arousal': arousal
        })

    # Process KEMDy18
    kemdy18_dir = os.path.join(base_path, 'KESDy18')
    if os.path.exists(kemdy18_dir):
        excel_files = glob.glob(os.path.join(kemdy18_dir, '*.xlsx'))
        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file)
                # First row (index 0) contains sub-headers, so data starts from index 1
                for idx in range(1, len(df)):
                    row = df.iloc[idx]
                    seg_id = row.iloc[1]
                    if pd.isna(seg_id):
                        continue
                    emotion = row.iloc[2]
                    valence = row.iloc[3]
                    arousal = row.iloc[4]
                    
                    # In KEMDy18, seg_id already contains '.wav'
                    add_to_metadata('KEMDy18', str(seg_id).strip(), emotion, valence, arousal)
            except Exception:
                pass

    # Process KEMDy19
    kemdy19_dir = os.path.join(base_path, 'KEMDy19_v1_4', 'annotation')
    if os.path.exists(kemdy19_dir):
        csv_files = glob.glob(os.path.join(kemdy19_dir, '*.csv'))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, skiprows=[1])
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    seg_id = row.iloc[9]
                    if pd.isna(seg_id):
                        continue
                    emotion = row.iloc[10]
                    valence = row.iloc[11]
                    arousal = row.iloc[12]

                    wav_name = f"{str(seg_id).strip()}.wav"
                    add_to_metadata('KEMDy19', wav_name, emotion, valence, arousal)
            except Exception:
                pass

    # Process KEMDy20
    kemdy20_dir = os.path.join(base_path, 'KEMDy20_v1_3', 'annotation')
    if os.path.exists(kemdy20_dir):
        csv_files = glob.glob(os.path.join(kemdy20_dir, '*.csv'))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, skiprows=[1])
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    seg_id = row.iloc[3]
                    if pd.isna(seg_id):
                        continue
                    emotion = row.iloc[4]
                    valence = row.iloc[5]
                    arousal = row.iloc[6]

                    wav_name = f"{str(seg_id).strip()}.wav"
                    add_to_metadata('KEMDy20', wav_name, emotion, valence, arousal)
            except Exception:
                pass

    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv_path = os.path.join(target_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_csv_path, index=False)

if __name__ == '__main__':
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    unified_dir = os.path.join(workspace_dir, 'Unified_KEMDy')
    process_datasets(workspace_dir, unified_dir)
