# coding: utf-8

# =============================================================================
# Determinism rate tables
# Standalone determinism breakdowns for Appendix D.7
# =============================================================================
#
# Input:  revised_master_v3.csv (576 rows)
# Output: determinism_by_model_dilemma.csv
#         determinism_by_model_language.csv
#         determinism_by_prompting.csv
#
# Dependencies: Python 3.x standard library
#
# This script was originally run in Google Colab with interactive file uploads.
# Please adapt the file I/O paths where it might be needed.
# =============================================================================


# =============================================================================
# Determinism rate tables
# =============================================================================

# =============================================================================
# Standalone determinism breakdowns for Appendix D.7
# =============================================================================
# Input: `revised_master_v3.csv` (576 rows)
# Output: 3 files in `output_determinism/`
# Dependencies: Python 3.x standard library


# =============================================================================
# Upload data
# =============================================================================


# [Colab upload cell — adapt file I/O for your environment]
# uploaded = files.upload()

# =============================================================================
# Imports and configuration
# =============================================================================


import csv
import os
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

# Input file: edit this path if your folder layout differs.
INPUT_FILE = '../05_Analysis/revised_master_v3.csv'
OUTPUT_DIR = 'output_determinism'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLM_NAME_MAP = {
    'gpt4o': 'GPT-4o', 'DeepSeek': 'DeepSeek R1',
    'LeChat Mistral Large': 'Mistral Large', 'Gemini 2.0 Flash Lite': 'Gemini Flash'
}
MODEL_ORDER = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini Flash']
DILEMMA_ORDER = ['MF', 'YO', 'HL']
LANGUAGE_ORDER = ['EN', 'CN', 'JP', 'ES']
PROMPTING_ORDER = ['PLAIN', 'CC', 'ZSCOT', 'FSCOT']

print(f'Output directory: {OUTPUT_DIR}/')


# =============================================================================
# Step 1: Load data
# =============================================================================
# File: revised_master_v3.csv. Determinism is defined as Fraction_A = 0.00 or 1.00.


# Step 1: Load data
# ============================================================

print("=" * 70)
print("Determinism rate computation")
print("Standalone tables for Appendix D.7")
print("=" * 70)
print(f"\nStep 1: Loading {INPUT_FILE}...")

with open(INPUT_FILE, 'r') as f:
    raw_rows = list(csv.DictReader(f))

rows = []
for r in raw_rows:
    fa = float(r['Fraction_A'])
    llm_std = r.get('LLM_std', LLM_NAME_MAP.get(r.get('LLM', ''), r.get('LLM', '')))
    rows.append({
        'Dilemma': r['Dilemma'],
        'Domain': r['Domain'],
        'Language': r['Language'],
        'Prompting': r['Prompting'],
        'LLM_std': llm_std,
        'Fraction_A': fa,
        'Is_Deterministic': 1 if (fa == 0.0 or fa == 1.0) else 0,
    })

total_det = sum(r['Is_Deterministic'] for r in rows)
print(f"  Loaded {len(rows)} rows")
print(f"  Deterministic cells (FA=0.0 or 1.0): {total_det}/{len(rows)} ({100*total_det/len(rows):.1f}%)")


# =============================================================================
# Step 2: Compute determinism by model × dilemma
# =============================================================================
# Cross-tabulation with row and column marginals + grand total.
# This is the primary determinism table (Appendix D.7), supporting the Section 7 argument that determinism varies by both model and dilemma.


# Step 2: Determinism by model × dilemma
# ============================================================


det_model_dilemma = []

# Per-cell: model × dilemma
for model in MODEL_ORDER:
    for dil in DILEMMA_ORDER:
        sub = [r for r in rows if r['LLM_std'] == model and r['Dilemma'] == dil]
        det = sum(r['Is_Deterministic'] for r in sub)
        n = len(sub)
        det_model_dilemma.append({
            'LLM_std': model, 'Dilemma': dil, 'N': n,
            'Deterministic_count': det,
            'Deterministic_pct': round(100 * det / n, 1)
        })

# Dilemma marginals (aggregated across models)
for dil in DILEMMA_ORDER:
    sub = [r for r in rows if r['Dilemma'] == dil]
    det = sum(r['Is_Deterministic'] for r in sub)
    det_model_dilemma.append({
        'LLM_std': 'Overall', 'Dilemma': dil, 'N': len(sub),
        'Deterministic_count': det,
        'Deterministic_pct': round(100 * det / len(sub), 1)
    })

# Model marginals (aggregated across dilemmas)
for model in MODEL_ORDER:
    sub = [r for r in rows if r['LLM_std'] == model]
    det = sum(r['Is_Deterministic'] for r in sub)
    det_model_dilemma.append({
        'LLM_std': model, 'Dilemma': 'Overall', 'N': len(sub),
        'Deterministic_count': det,
        'Deterministic_pct': round(100 * det / len(sub), 1)
    })

# Grand total
det_model_dilemma.append({
    'LLM_std': 'Overall', 'Dilemma': 'Overall', 'N': len(rows),
    'Deterministic_count': total_det,
    'Deterministic_pct': round(100 * total_det / len(rows), 1)
})

outpath = os.path.join(OUTPUT_DIR, 'determinism_by_model_dilemma.csv')
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['LLM_std', 'Dilemma', 'N', 'Deterministic_count', 'Deterministic_pct'])
    w.writeheader(); w.writerows(det_model_dilemma)
print(f"  Written: determinism_by_model_dilemma.csv ({len(det_model_dilemma)} rows)")

# Display
print(f"\n  {'Model':<18} {'Dilemma':<8} {'N':>4} {'Det':>4} {'Det%':>6}")
print(f"  {'-'*45}")
for r in det_model_dilemma:
    print(f"  {r['LLM_std']:<18} {r['Dilemma']:<8} {r['N']:>4} {r['Deterministic_count']:>4} {r['Deterministic_pct']:>5.1f}%")


# =============================================================================
# Step 3: Compute determinism by model × language
# =============================================================================
# Cross-tabulation with marginals.


# Step 3: Determinism by model × language
# ============================================================


det_model_lang = []

# Per-cell: model × language
for model in MODEL_ORDER:
    for lang in LANGUAGE_ORDER:
        sub = [r for r in rows if r['LLM_std'] == model and r['Language'] == lang]
        det = sum(r['Is_Deterministic'] for r in sub)
        n = len(sub)
        det_model_lang.append({
            'LLM_std': model, 'Language': lang, 'N': n,
            'Deterministic_count': det,
            'Deterministic_pct': round(100 * det / n, 1)
        })

# Language marginals (aggregated across models)
for lang in LANGUAGE_ORDER:
    sub = [r for r in rows if r['Language'] == lang]
    det = sum(r['Is_Deterministic'] for r in sub)
    det_model_lang.append({
        'LLM_std': 'Overall', 'Language': lang, 'N': len(sub),
        'Deterministic_count': det,
        'Deterministic_pct': round(100 * det / len(sub), 1)
    })

# Model marginals (aggregated across languages)
for model in MODEL_ORDER:
    sub = [r for r in rows if r['LLM_std'] == model]
    det = sum(r['Is_Deterministic'] for r in sub)
    det_model_lang.append({
        'LLM_std': model, 'Language': 'Overall', 'N': len(sub),
        'Deterministic_count': det,
        'Deterministic_pct': round(100 * det / len(sub), 1)
    })

# Grand total
det_model_lang.append({
    'LLM_std': 'Overall', 'Language': 'Overall', 'N': len(rows),
    'Deterministic_count': total_det,
    'Deterministic_pct': round(100 * total_det / len(rows), 1)
})

outpath = os.path.join(OUTPUT_DIR, 'determinism_by_model_language.csv')
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['LLM_std', 'Language', 'N', 'Deterministic_count', 'Deterministic_pct'])
    w.writeheader(); w.writerows(det_model_lang)
print(f"  Written: determinism_by_model_language.csv ({len(det_model_lang)} rows)")

# Display
print(f"\n  {'Model':<18} {'Lang':<8} {'N':>4} {'Det':>4} {'Det%':>6}")
print(f"  {'-'*45}")
for r in det_model_lang:
    print(f"  {r['LLM_std']:<18} {r['Language']:<8} {r['N']:>4} {r['Deterministic_count']:>4} {r['Deterministic_pct']:>5.1f}%")


# =============================================================================
# Step 4: Compute determinism by prompting
# =============================================================================
# Aggregation across models, dilemmas, domains, and languages.


# Step 4: Determinism by prompting regimes
# ============================================================


det_prompting = []

for p in PROMPTING_ORDER:
    sub = [r for r in rows if r['Prompting'] == p]
    det = sum(r['Is_Deterministic'] for r in sub)
    n = len(sub)
    det_prompting.append({
        'Prompting': p, 'N': n,
        'Deterministic_count': det,
        'Deterministic_pct': round(100 * det / n, 1)
    })

outpath = os.path.join(OUTPUT_DIR, 'determinism_by_prompting.csv')
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Prompting', 'N', 'Deterministic_count', 'Deterministic_pct'])
    w.writeheader(); w.writerows(det_prompting)
print(f"  Written: determinism_by_prompting.csv ({len(det_prompting)} rows)")

# Display
print(f"\n  {'Prompting':<10} {'N':>4} {'Det':>4} {'Det%':>6}")
print(f"  {'-'*30}")
for r in det_prompting:
    print(f"  {r['Prompting']:<10} {r['N']:>4} {r['Deterministic_count']:>4} {r['Deterministic_pct']:>5.1f}%")


# =============================================================================
# Step 5: Verification
# =============================================================================
# Cross-check all output values against the paper.


# Step 5: Verification
# ============================================================


rpt = []; pc = fc = 0
def V(cond, desc, exp, act):
    global pc, fc
    if cond: pc += 1; rpt.append(f"  ✓ {desc}: {act}")
    else: fc += 1; rpt.append(f"  ✗ FAIL {desc}: expected={exp}, actual={act}")

# Helper lookups
def md(model, dil):
    return [r for r in det_model_dilemma if r['LLM_std']==model and r['Dilemma']==dil][0]
def ml(model, lang):
    return [r for r in det_model_lang if r['LLM_std']==model and r['Language']==lang][0]
def dp(prompt):
    return [r for r in det_prompting if r['Prompting']==prompt][0]

rpt.append("\n--- Overall ---")
V(total_det == 302, "Total deterministic cells", 302, total_det)
V(round(100*total_det/576, 1) == 52.4, "Overall det %", 52.4, round(100*total_det/576, 1))

rpt.append("\n--- Model × Dilemma (paper Section 7 claims) ---")
V(md('GPT-4o','MF')['Deterministic_pct'] == 87.5, "GPT-4o MF", 87.5, md('GPT-4o','MF')['Deterministic_pct'])
V(md('GPT-4o','YO')['Deterministic_pct'] == 43.8, "GPT-4o YO", 43.8, md('GPT-4o','YO')['Deterministic_pct'])
V(md('GPT-4o','HL')['Deterministic_pct'] == 60.4, "GPT-4o HL", 60.4, md('GPT-4o','HL')['Deterministic_pct'])
V(md('Mistral Large','Overall')['Deterministic_pct'] == 32.6, "Mistral overall", 32.6, md('Mistral Large','Overall')['Deterministic_pct'])
V(md('GPT-4o','Overall')['Deterministic_pct'] == 63.9, "GPT-4o overall", 63.9, md('GPT-4o','Overall')['Deterministic_pct'])
V(md('DeepSeek R1','Overall')['Deterministic_pct'] == 60.4, "DeepSeek overall", 60.4, md('DeepSeek R1','Overall')['Deterministic_pct'])
V(md('Gemini Flash','Overall')['Deterministic_pct'] == 52.8, "Gemini overall", 52.8, md('Gemini Flash','Overall')['Deterministic_pct'])

rpt.append("\n--- Dilemma marginals ---")
V(md('Overall','MF')['Deterministic_pct'] == 70.3, "MF overall", 70.3, md('Overall','MF')['Deterministic_pct'])
V(md('Overall','YO')['Deterministic_pct'] == 40.1, "YO overall", 40.1, md('Overall','YO')['Deterministic_pct'])
V(md('Overall','HL')['Deterministic_pct'] == 46.9, "HL overall", 46.9, md('Overall','HL')['Deterministic_pct'])

rpt.append("\n--- Prompting (paper Section 6.5 claims) ---")
V(dp('PLAIN')['Deterministic_pct'] == 59.0, "PLAIN det", 59.0, dp('PLAIN')['Deterministic_pct'])
V(dp('CC')['Deterministic_pct'] == 66.7, "CC det", 66.7, dp('CC')['Deterministic_pct'])
V(dp('ZSCOT')['Deterministic_pct'] == 27.1, "ZSCoT det", 27.1, dp('ZSCOT')['Deterministic_pct'])
V(dp('FSCOT')['Deterministic_pct'] == 56.9, "FSCoT det", 56.9, dp('FSCOT')['Deterministic_pct'])

rpt.append("\n--- Lowest / highest extremes ---")
V(md('Mistral Large','YO')['Deterministic_pct'] == 22.9, "Lowest: Mistral YO", 22.9, md('Mistral Large','YO')['Deterministic_pct'])
V(md('GPT-4o','MF')['Deterministic_pct'] == 87.5, "Highest: GPT-4o MF", 87.5, md('GPT-4o','MF')['Deterministic_pct'])

rpt.append("\n--- Option B determinism (positional bias defence) ---")
b_det = sum(1 for r in rows if r['Is_Deterministic'] == 1 and r['Fraction_A'] == 0.0)
V(b_det == 64, "FA=0.0 deterministic cells", 64, b_det)

print('\n'.join(rpt))
print(f"\n{'='*70}")
print(f"Passed: {pc}  |  Failed: {fc}")
if fc == 0: print("All checks passed. ✓")
else: print("WARNING: Some checks failed. Review above.")


# =============================================================================
# Step 6: Package outputs
# =============================================================================
# Zip all output files for download.


# [Colab download cell — adapt for your environment]
# files.download(zip_path)