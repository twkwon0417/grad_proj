import pandas as pd
import os

def analyze_statistics():
    # 메타데이터 로드
    metadata_path = os.path.join(os.path.dirname(__file__), 'Unified_KEMDy', 'metadata.csv')
    df = pd.read_csv(metadata_path)

    # 마크다운 시작
    md_content = """# Unified KEMDy Dataset Statistics

## Overview
"""

    # 전체 통계
    total_files = len(df)
    md_content += f"""- **Total Audio Files**: {total_files}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Distribution
"""

    dataset_counts = df['dataset_source'].value_counts().sort_index()
    md_content += "\n| Dataset | Count | Percentage |\n"
    md_content += "|---------|-------|------------|\n"
    for dataset, count in dataset_counts.items():
        percentage = (count / total_files) * 100
        md_content += f"| {dataset} | {count} | {percentage:.2f}% |\n"

    # 감정 분포
    md_content += "\n## 2. Emotion Distribution\n"
    emotion_counts = df['Emotion'].value_counts().sort_values(ascending=False)
    md_content += "\n| Emotion | Count | Percentage |\n"
    md_content += "|---------|-------|------------|\n"
    for emotion, count in emotion_counts.items():
        percentage = (count / total_files) * 100
        md_content += f"| {emotion} | {count} | {percentage:.2f}% |\n"

    # 데이터셋별 감정 분포
    md_content += "\n### Emotion Distribution by Dataset\n"
    emotion_by_dataset = pd.crosstab(df['dataset_source'], df['Emotion'])
    md_content += "\n" + emotion_by_dataset.to_markdown() + "\n"

    # Valence 통계
    md_content += "\n## 3. Valence Statistics\n"
    valence_stats = df['Valence'].describe()
    md_content += f"""
- **Mean**: {valence_stats['mean']:.4f}
- **Std Dev**: {valence_stats['std']:.4f}
- **Min**: {valence_stats['min']:.4f}
- **25%**: {valence_stats['25%']:.4f}
- **Median (50%)**: {valence_stats['50%']:.4f}
- **75%**: {valence_stats['75%']:.4f}
- **Max**: {valence_stats['max']:.4f}
"""

    # Arousal 통계
    md_content += "\n## 4. Arousal Statistics\n"
    arousal_stats = df['Arousal'].describe()
    md_content += f"""
- **Mean**: {arousal_stats['mean']:.4f}
- **Std Dev**: {arousal_stats['std']:.4f}
- **Min**: {arousal_stats['min']:.4f}
- **25%**: {arousal_stats['25%']:.4f}
- **Median (50%)**: {arousal_stats['50%']:.4f}
- **75%**: {arousal_stats['75%']:.4f}
- **Max**: {arousal_stats['max']:.4f}
"""

    # 감정별 Valence/Arousal 평균
    md_content += "\n## 5. Valence & Arousal by Emotion\n"
    
    md_content += "\n### Valence by Emotion\n"
    valence_by_emotion = df.groupby('Emotion')['Valence'].agg(['mean', 'std', 'min', 'max']).round(4)
    md_content += "\n" + valence_by_emotion.to_markdown() + "\n"

    md_content += "\n### Arousal by Emotion\n"
    arousal_by_emotion = df.groupby('Emotion')['Arousal'].agg(['mean', 'std', 'min', 'max']).round(4)
    md_content += "\n" + arousal_by_emotion.to_markdown() + "\n"

    # 데이터셋별 통계
    md_content += "\n## 6. Statistics by Dataset\n"
    for dataset in sorted(df['dataset_source'].unique()):
        dataset_df = df[df['dataset_source'] == dataset]
        md_content += f"\n### {dataset}\n"
        md_content += f"- **Total Files**: {len(dataset_df)}\n"
        md_content += f"- **Emotions**: {', '.join(sorted(dataset_df['Emotion'].unique()))}\n"
        md_content += f"- **Valence Mean**: {dataset_df['Valence'].mean():.4f} ± {dataset_df['Valence'].std():.4f}\n"
        md_content += f"- **Arousal Mean**: {dataset_df['Arousal'].mean():.4f} ± {dataset_df['Arousal'].std():.4f}\n"

    # 파일 저장
    output_path = os.path.join(os.path.dirname(__file__), 'Unified_KEMDy', 'STATISTICS.md')
    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"✅ 통계 분석 완료!")
    print(f"📄 저장 경로: {output_path}")
    print("\n" + "="*50)
    print(md_content)

if __name__ == '__main__':
    analyze_statistics()
