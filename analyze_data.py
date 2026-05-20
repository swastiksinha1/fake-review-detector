import pandas as pd
import numpy as np

df = pd.read_csv('data/reviews.csv')
df = df.rename(columns={'text_': 'text'})
df = df[['text', 'label', 'rating']].dropna()
df['label'] = df['label'].str.strip().str.upper()
df['label'] = df['label'].map({'OR': 'real', 'CG': 'fake'})
df = df.dropna(subset=['label'])

print("=== RATING vs LABEL CROSSTAB ===")
ct = pd.crosstab(df['rating'], df['label'], margins=True)
print(ct)

print("\n=== RATING vs LABEL (row percentages) ===")
ct_pct = pd.crosstab(df['rating'], df['label'], normalize='index')
print(ct_pct.round(3))

print("\n=== MEAN RATING by LABEL ===")
print(df.groupby('label')['rating'].describe())

print("\n=== TEXT LENGTH STATS by LABEL ===")
df['word_count'] = df['text'].str.split().str.len()
print(df.groupby('label')['word_count'].describe().round(1))

for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
    subset = df[df['rating'] == star]
    real_count = (subset['label'] == 'real').sum()
    fake_count = (subset['label'] == 'fake').sum()
    real_pct = real_count / len(subset) * 100 if len(subset) > 0 else 0
    fake_pct = fake_count / len(subset) * 100 if len(subset) > 0 else 0
    print(f"\n=== {int(star)}-STAR REVIEWS ===")
    print(f"  Total: {len(subset)}")
    print(f"  Real: {real_count} ({real_pct:.1f}%)")
    print(f"  Fake: {fake_count} ({fake_pct:.1f}%)")

# Check category distribution
df2 = pd.read_csv('data/reviews.csv')
print("\n=== CATEGORY DISTRIBUTION ===")
print(df2['category'].value_counts())
