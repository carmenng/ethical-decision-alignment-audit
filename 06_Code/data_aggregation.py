# coding: utf-8

# =============================================================================
# Data aggregation pipeline
# Raw LLM decision logs > per-dilemma master summaries > combined summary
# =============================================================================
#
# Input:  Raw CSV files from 02_Raw Data/ (1,200 rows each: 12 scenarios × 100 runs)
# Output: MF_master_summary.csv (192 rows)
#         YO_master_summary.csv (192 rows)
#         HL_master_summary.csv (192 rows)
#         combined_master_summary.csv (576 rows)
#
# Libraries: pandas, numpy, scipy
#
# Each raw file covers one model × one prompting regime
# (12 scenarios = 3 domains × 4 languages, 100 runs each).
# There are 16 raw files per dilemma (4 models × 4 prompting regimes) = 48 files total.
#
# This script was originally run in Google Colab with interactive file uploads.
# Please feel free to adapt the file I/O paths if needed.
# =============================================================================


# =============================================================================
# Data aggregation pipeline
# =============================================================================

# =============================================================================
# Raw LLM decision logs > per-dilemma master summaries > combined summary
# =============================================================================
# **Input:** Raw CSV files from `02_Raw Data/` (1,200 rows each: 12 scenarios × 100 runs)
# **Output:**
# - `MF_master_summary.csv` (192 rows)
# - `YO_master_summary.csv` (192 rows)
# - `HL_master_summary.csv` (192 rows)
# - `combined_master_summary.csv` (576 rows)
# Could need: pandas, numpy, scipy
# Each raw file covers one model × one prompting regime (12 scenarios = 3 domains × 4 languages, 100 runs each).
# There are 16 raw files per dilemma (4 models × 4 prompting regimes) = 48 files total.
# Process one dilemma at a time (MF, then YO, then HL), or all at once.
# Results will be first analysis of Fraction_A, SD, P-value etc.


# =============================================================================
# Config
# =============================================================================


import pandas as pd
import numpy as np
from scipy.stats import norm
import csv
import os
import re
import io

OUTPUT_DIR = 'output_summaries'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MME_Score per country-language condition per dilemma (portal min-max normalized AMCEs)
# Already verified against OSF CountriesChangePr.csv in Appendix A.3
MME_SCORES = {
    'MF': {'EN': 0.68, 'CN': 0.21, 'JP': 0.00, 'ES': 0.53},
    'YO': {'EN': 0.63, 'CN': 0.10, 'JP': 0.28, 'ES': 0.68},
    'HL': {'EN': 0.52, 'CN': 0.43, 'JP': 0.37, 'ES': 0.63},
}

# Map filename LLM tokens to canonical names used throughout
LLM_FROM_FILENAME = {
    'gpt4o': 'gpt4o',
    'deepseek': 'DeepSeek',
    'mistral_large': 'LeChat Mistral Large',
    'gemini': 'Gemini 2.0 Flash Lite',
}

LANGUAGE_ORDER = ['EN', 'CN', 'JP', 'ES']
DOMAIN_ORDER = ['D1', 'D2', 'D3']
PROMPTING_ORDER = ['PLAIN', 'CC', 'ZSCOT', 'FSCOT']
LLM_ORDER = ['gpt4o', 'DeepSeek', 'LeChat Mistral Large', 'Gemini 2.0 Flash Lite']


# =============================================================================
# Stage 1: Raw decision logs >> per-dilemma master summary
# =============================================================================
# Each raw CSV has columns: `scenario_id, language, run_number, decision, explanation`
# Filenames follow: `[DILEMMA]_[PROMPTING]_results_[llm].csv`
# For each file we are doing these steps:
# 1. Parse dilemma, prompting regime, and LLM from the filename
# 2. Group the 1,200 rows by scenario_id (12 groups × 100 runs each)
# 3. Compute Fraction_A = count(decision == 'A') / 100
# 4. Compute SD as binomial standard error: sqrt(p(1-p)/n), or 0 if deterministic
# 5. Compute 95% CI bounds, clamped to [0, 1]
# 6. Compute p-value testing Fraction_A against the MME_Score benchmark
# 7. Assign the correct MME_Score by language


def parse_filename(filename):
    """Extract dilemma, prompting, and LLM from raw CSV filename.
    Expected format: [DILEMMA]_[PROMPTING]_results_[llm].csv
    """
    # Strip path and any Colab duplicate suffix like ' (1)'
    base = os.path.basename(filename)
    base = re.sub(r'\s*\(\d+\)', '', base).replace('.csv', '')

    parts = base.split('_')
    dilemma = parts[0]  # MF, YO, or HL
    prompting = parts[1]  # PLAIN, CC, ZSCOT, FSCOT

    # LLM token is everything after 'results_'
    llm_match = re.search(r'results_(.+)$', base)
    if not llm_match:
        raise ValueError(f"Cannot parse LLM from filename: {filename}")
    llm_token = llm_match.group(1).strip().lower()

    # Match against known LLM tokens
    llm_name = None
    for key, name in LLM_FROM_FILENAME.items():
        if key.lower() in llm_token:
            llm_name = name
            break
    if llm_name is None:
        raise ValueError(f"Unrecognized LLM token '{llm_token}' in {filename}")

    return dilemma, prompting, llm_name


def compute_p_value(fraction_a, mme_score, n=100):
    """Two-sided z-test: does Fraction_A differ from MME_Score?"""
    if fraction_a == mme_score:
        return 1.0
    # Deterministic case
    if fraction_a in (0.0, 1.0):
        return 0.0
    se = np.sqrt(mme_score * (1 - mme_score) / n) if 0 < mme_score < 1 else 0
    if se == 0:
        return 0.0 if fraction_a != mme_score else 1.0
    z = (fraction_a - mme_score) / se
    return min(2 * (1 - norm.cdf(abs(z))), 1.0)


def process_raw_file(filepath, content=None):
    """Process one raw CSV → 12-row summary (3 domains × 4 languages)."""
    dilemma, prompting, llm = parse_filename(filepath)
    mme_scores = MME_SCORES[dilemma]

    if content is not None:
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_csv(filepath)

    # Validate
    assert set(df.columns) >= {'scenario_id', 'language', 'run_number', 'decision'}, \
        f"Missing columns in {filepath}"
    assert df['decision'].isin(['A', 'B']).all(), \
        f"Invalid decisions in {filepath}: {df['decision'].unique()}"

    # Extract domain and language from scenario_id
    df['Domain'] = df['scenario_id'].str.split('_').str[0]
    df['Language'] = df['scenario_id'].str[-2:]

    results = []
    for sid, group in df.groupby('scenario_id'):
        n = len(group)
        fa = (group['decision'] == 'A').mean()
        sd = np.sqrt(fa * (1 - fa) / n) if fa not in (0.0, 1.0) else 0.0
        ci_lo = max(fa - 1.96 * sd, 0.0)
        ci_hi = min(fa + 1.96 * sd, 1.0)
        lang = group['Language'].iloc[0]
        domain = group['Domain'].iloc[0]
        mme = mme_scores[lang]
        pval = compute_p_value(fa, mme, n)

        results.append({
            'Dilemma': dilemma, 'Domain': domain, 'Language': lang,
            'Prompting': prompting, 'LLM': llm,
            'Scenario_ID': sid, 'MME_Score': mme,
            'Fraction_A': round(fa, 2), 'SD': round(sd, 2),
            'CI_Lower': round(ci_lo, 2), 'CI_Upper': round(ci_hi, 2),
            'P_Value': round(pval, 2),
        })

    summary = pd.DataFrame(results)
    summary['Domain'] = pd.Categorical(summary['Domain'], categories=DOMAIN_ORDER, ordered=True)
    summary['Language'] = pd.Categorical(summary['Language'], categories=LANGUAGE_ORDER, ordered=True)
    summary = summary.sort_values(['Domain', 'Language']).reset_index(drop=True)

    assert len(summary) == 12, f"Expected 12 rows, got {len(summary)} for {filepath}"
    return summary


# =============================================================================
# Upload and process raw files
# =============================================================================
# For Upload raw CSV files in four batches (one per prompting regime folder).
# Each batch expects 12 files (3 dilemmas × 4 models).
# Run all four upload cells, then proceed to the next section.


# [Colab upload cell — adapt file I/O for your environment]
# uploaded = files.upload()

process_batch('PLAIN')


process_batch('CC')


process_batch('ZSCOT')


process_batch('FSCOT')


# Summary after all four batches
print(f"\nTotal files uploaded: {len(all_uploaded)}")
if errors:
    print(f"{len(errors)} files failed:")
    for fn, err in errors:
        print(f"  {fn}: {err}")
else:
    print("All files processed successfully.")

for dil in ['MF', 'YO', 'HL']:
    n_files = len(dilemma_frames[dil])
    n_rows = sum(len(df) for df in dilemma_frames[dil])
    expected = 16  # 4 models × 4 prompting regimes
    status = '✓' if n_files == expected else f'✗ expected {expected}, got {n_files}'
    print(f"  {dil}: {n_files} files, {n_rows} rows {status}")


# =============================================================================
# Write per-dilemma master summaries
# =============================================================================


column_order = ['Dilemma', 'Domain', 'Language', 'Prompting', 'LLM',
                'Scenario_ID', 'MME_Score', 'Fraction_A', 'SD',
                'CI_Lower', 'CI_Upper', 'P_Value']

for dil in ['MF', 'YO', 'HL']:
    if not dilemma_frames[dil]:
        print(f"  {dil}: no data, skipping")
        continue

    master = pd.concat(dilemma_frames[dil], ignore_index=True)

    # Sort by prompting → LLM → domain → language
    master['Prompting'] = pd.Categorical(master['Prompting'], categories=PROMPTING_ORDER, ordered=True)
    master['LLM'] = pd.Categorical(master['LLM'], categories=LLM_ORDER, ordered=True)
    master['Domain'] = pd.Categorical(master['Domain'], categories=DOMAIN_ORDER, ordered=True)
    master['Language'] = pd.Categorical(master['Language'], categories=LANGUAGE_ORDER, ordered=True)
    master = master.sort_values(['Prompting', 'LLM', 'Domain', 'Language']).reset_index(drop=True)
    master = master[column_order]

    outpath = os.path.join(OUTPUT_DIR, f'{dil}_master_summary.csv')
    master.to_csv(outpath, index=False)
    print(f"  {outpath}: {len(master)} rows")

    # Spot check: show Fraction_A range
    print(f"    Fraction_A range: {master['Fraction_A'].min()} – {master['Fraction_A'].max()}")
    print(f"    Deterministic (FA=0 or 1): {((master['Fraction_A']==0)|(master['Fraction_A']==1)).sum()}/192")


# =============================================================================
# Stage 2: Combine three dilemma summaries into one overall file
# =============================================================================
# No new columns added, only grouping summaries.
# This is the input for the analysis pipeline.


frames = []
for dil in ['MF', 'YO', 'HL']:
    path = os.path.join(OUTPUT_DIR, f'{dil}_master_summary.csv')
    df = pd.read_csv(path)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
outpath = os.path.join(OUTPUT_DIR, 'combined_master_summary.csv')
combined.to_csv(outpath, index=False)
print(f"{outpath}: {len(combined)} rows, {len(combined.columns)} columns")

# Verify dilemma counts
for dil in ['MF', 'YO', 'HL']:
    n = len(combined[combined['Dilemma'] == dil])
    print(f"  {dil}: {n} rows {'✓' if n == 192 else '✗'}")


# =============================================================================
# Verification
# =============================================================================
# Cross-check key values.


pc = fc = 0
def V(cond, desc, exp, act):
    global pc, fc
    if cond: pc += 1; print(f"  ✓ {desc}: {act}")
    else: fc += 1; print(f"  ✗ {desc}: expected {exp}, got {act}")

V(len(combined) == 576, "Total rows", 576, len(combined))

# MME_Score correctness: one value per language per dilemma
for dil in ['MF', 'YO', 'HL']:
    for lang in LANGUAGE_ORDER:
        vals = combined[(combined['Dilemma']==dil) & (combined['Language']==lang)]['MME_Score'].unique()
        expected = MME_SCORES[dil][lang]
        V(len(vals)==1 and vals[0]==expected, f"{dil} {lang} MME_Score", expected, vals)

# Determinism count (FA = 0.0 or 1.0)
det = ((combined['Fraction_A']==0.0) | (combined['Fraction_A']==1.0)).sum()
V(det == 302, "Deterministic cells", 302, det)

# Spot checks from paper
gpt_mf_en_plain = combined[(combined['LLM']=='gpt4o') & (combined['Dilemma']=='MF') &
                           (combined['Language']=='EN') & (combined['Prompting']=='PLAIN')]
V((gpt_mf_en_plain['Fraction_A']==1.0).all(), "GPT-4o MF EN PLAIN all FA=1.0", True,
   (gpt_mf_en_plain['Fraction_A']==1.0).all())

mistral_mf_en_plain = combined[(combined['LLM']=='LeChat Mistral Large') & (combined['Dilemma']=='MF') &
                               (combined['Language']=='EN') & (combined['Prompting']=='PLAIN') &
                               (combined['Domain']=='D1')]
V(mistral_mf_en_plain['Fraction_A'].values[0] == 0.58,
  "Mistral MF EN PLAIN D1 FA", 0.58, mistral_mf_en_plain['Fraction_A'].values[0])

print(f"\nPassed: {pc} | Failed: {fc}")
if fc == 0: print("All checks passed.")


# =============================================================================
# Download outputs
# =============================================================================


# [Colab download cell — adapt for your environment]
# files.download(zip_path)