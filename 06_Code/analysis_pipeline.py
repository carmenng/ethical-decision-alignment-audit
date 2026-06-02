# coding: utf-8

# =============================================================================
# Analysis pipeline v3
# Layer 1 (gradient concordance) + Layer 2 (governance typology)
# + Positional bias stress test (both layers)
# =============================================================================
#
# Input:  combined_master_summary.csv (576 rows, 12 columns)
# Output: revised_master_v3.csv + tau tables + typology summaries
#         + positional bias tables + spearman + verification report
#
# Dependencies: Python 3.x standard library
#
# Originally run in Google Colab. For standalone use, please set INPUT_FILE
# to your local path and adapt output paths as needed.
# =============================================================================


# =============================================================================
# Analysis pipeline
# =============================================================================
# ## Layer 1 (gradient concordance) + Layer 2 (governance typology) + Positional bias
# Input: combined_master_summary.csv (576 rows)
# Output: 27 files in `output_v3/`

# ######## Upload data

# ######## Imports, configuration, and equations


import csv
import math
import os
import json
import zipfile
from collections import defaultdict, Counter

# Input file: edit this path if your folder layout differs.
INPUT_FILE = '../04_Combined_Summary/combined_master_summary.csv'
OUTPUT_DIR = 'output_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLM_NAME_MAP = {
    'gpt4o': 'GPT-4o', 'DeepSeek': 'DeepSeek R1',
    'LeChat Mistral Large': 'Mistral Large', 'Gemini 2.0 Flash Lite': 'Gemini Flash'
}
MODEL_ORDER = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini Flash']
LANG_COUNTRY_MAP = {'EN': 'USA', 'CN': 'China', 'JP': 'Japan', 'ES': 'Mexico'}
BIN_NAMES = {
    1: 'Calibrated', 2: 'Rigid tracking', 3: 'Gradient-sensitive overshoot',
    4: 'Gradient erased', 5: 'Gradient inverted',
    6: 'Non-tracking contradiction', 7: 'Non-tracking rigidity'
}
POSBIAS_DELTAS = [0.05, 0.10]

# Equations

def kendall_tau(fa_values, mme_values):
    """EQ1 (Layer 1): tau = (C - D) / (C + D + T)."""
    c, d, t = 0, 0, 0
    for i in range(len(fa_values)):
        for j in range(i + 1, len(fa_values)):
            md = mme_values[i] - mme_values[j]
            fd = fa_values[i] - fa_values[j]
            if md == 0 or fd == 0: t += 1
            elif (md > 0 and fd > 0) or (md < 0 and fd < 0): c += 1
            else: d += 1
    total = c + d + t
    return (round((c-d)/total, 4) if total > 0 else 0.0), c, d, t

def spearman_rho(x_vals, y_vals):
    """EQ2 (Layer 1 supplementary): Spearman's rho = Pearson on ranks."""
    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]: j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j): ranks[indexed[k][0]] = avg
            i = j
        return ranks
    def _pearson(x, y):
        n = len(x)
        if n < 3: return None
        mx, my = sum(x)/n, sum(y)/n
        sx = math.sqrt(sum((xi-mx)**2 for xi in x)/(n-1))
        sy = math.sqrt(sum((yi-my)**2 for yi in y)/(n-1))
        if sx == 0 or sy == 0: return None
        return round(sum((x[i]-mx)*(y[i]-my) for i in range(n))/(n-1)/(sx*sy), 4)
    return _pearson(_rank(x_vals), _rank(y_vals))

def local_concordance(fa_i, mme_i, other_fas, other_mmes):
    """EQ3 (Layer 2, dimension 1): LC = concordant - discordant. Range -3 to +3."""
    c, d = 0, 0
    for fa_j, mme_j in zip(other_fas, other_mmes):
        p = (fa_i - fa_j) * (mme_i - mme_j)
        if p > 0: c += 1
        elif p < 0: d += 1
    return c - d

def gradient_fit(lc):
    """EQ4 (Layer 2): Tracking / Undifferentiated / Inverting."""
    return 'Tracking' if lc > 0 else ('Undifferentiated' if lc == 0 else 'Inverting')

def direction(fa):
    """EQ5 (Layer 2, dimension 2): A-territory (FA >= 0.50) / B-territory (FA < 0.50)."""
    return 'A-territory' if fa >= 0.50 else 'B-territory'

def deliberation(fa):
    """EQ6 (Layer 2, dimension 3): Deterministic / Near-deterministic / Variable."""
    if fa == 0.0 or fa == 1.0: return 'Deterministic'
    elif fa >= 0.95 or fa <= 0.05: return 'Near-deterministic'
    else: return 'Variable'

def del_group(delib):
    """EQ7 (Layer 2): Grouped deliberation for binning."""
    return 'Variable' if delib == 'Variable' else 'Det/Near-det'

def assign_cell_type(gf, dirn, dg):
    """EQ8 (Layer 2): 7-bin governance typology."""
    if gf == 'Tracking' and dirn == 'A-territory' and dg == 'Variable': return 1, 'Calibrated'
    elif gf == 'Tracking' and dirn == 'A-territory' and dg == 'Det/Near-det': return 2, 'Rigid tracking'
    elif gf == 'Tracking' and dirn == 'B-territory': return 3, 'Gradient-sensitive overshoot'
    elif gf == 'Undifferentiated': return 4, 'Gradient erased'
    elif gf == 'Inverting' and dirn == 'A-territory': return 5, 'Gradient inverted'
    elif gf != 'Tracking' and dirn == 'B-territory' and dg == 'Variable': return 6, 'Non-tracking contradiction'
    elif gf != 'Tracking' and dirn == 'B-territory' and dg == 'Det/Near-det': return 7, 'Non-tracking rigidity'
    return 0, 'UNCLASSIFIED'

def compute_full_typology(fas, mmes, condition_indices_map):
    """Compute all Layer 2 columns for a set of FA values (original or adjusted).
    Returns list of dicts with LC, Gradient_Fit, Direction, Deliberation, Del_Group, Cell_Type, Bin_Score."""
    results = [None] * len(fas)
    for key, indices in condition_indices_map.items():
        for idx in indices:
            ofas = [fas[j] for j in indices if j != idx]
            ommes = [mmes[j] for j in indices if j != idx]
            lc = local_concordance(fas[idx], mmes[idx], ofas, ommes)
            gf = gradient_fit(lc)
            dirn = direction(fas[idx])
            delib = deliberation(fas[idx])
            dg = del_group(delib)
            bs, ct = assign_cell_type(gf, dirn, dg)
            results[idx] = {'LC': lc, 'Gradient_Fit': gf, 'Direction': dirn,
                           'Deliberation': delib, 'Del_Group': dg, 'Cell_Type': ct, 'Bin_Score': bs}
    return results

# ######## Step 1: Load data
# ######## Read CSV, extract columns, group by condition.


# Step 1: Load data

print("=" * 70)
print("=" * 70)

with open(INPUT_FILE, 'r') as f:
    raw_rows = list(csv.DictReader(f))

rows = []
for r in raw_rows:
    rows.append({
        'Dilemma': r['Dilemma'], 'Domain': r['Domain'], 'Language': r['Language'],
        'Country': LANG_COUNTRY_MAP.get(r['Language'], r['Language']),
        'Prompting': r['Prompting'], 'LLM': r['LLM'],
        'LLM_std': LLM_NAME_MAP.get(r['LLM'], r['LLM']),
        'Scenario_ID': r['Scenario_ID'],
        'MME_Score': float(r['MME_Score']), 'Fraction_A': float(r['Fraction_A']),
        'SD': float(r['SD']), 'CI_Lower': float(r['CI_Lower']),
        'CI_Upper': float(r['CI_Upper']), 'P_Value': float(r['P_Value']),
        'Is_Deterministic': 1 if (float(r['Fraction_A']) == 0.0 or float(r['Fraction_A']) == 1.0) else 0,
    })

condition_groups = defaultdict(list)
for i, r in enumerate(rows):
    condition_groups[(r['LLM'], r['Dilemma'], r['Domain'], r['Prompting'])].append(i)

print(f"  {len(rows)} cells, {len(condition_groups)} conditions")

# ######## Step 2: Layer 1 — Kendall's tau per condition
# ######## Does the LLM rank countries in the documented order?


# Step 2: Layer 1 — Does the LLM rank countries in the documented order?
#
# Kendall's tau: scale-free rank concordance. Compares only orderings.
# 4 countries per condition = 6 pairs. 144 conditions = 864 pairs.

tau_conditions = []
for key, indices in sorted(condition_groups.items()):
    group = sorted([rows[i] for i in indices], key=lambda x: x['Language'])
    fa = [r['Fraction_A'] for r in group]
    mme = [r['MME_Score'] for r in group]
    langs = [r['Language'] for r in group]
    tau, c, d, t = kendall_tau(fa, mme)
    tau_conditions.append({
        'LLM': LLM_NAME_MAP.get(key[0], key[0]), 'Dilemma': key[1],
        'Domain': key[2], 'Prompting': key[3],
        'Tau': tau, 'Concordant': c, 'Discordant': d, 'Tied': t, 'N_pairs': c+d+t,
        'FA_values': '|'.join(f"{l}={v}" for l, v in zip(langs, fa)),
        'MME_values': '|'.join(f"{l}={v}" for l, v in zip(langs, mme)),
    })

with open(os.path.join(OUTPUT_DIR, 'tau_per_condition.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(tau_conditions[0].keys()))
    w.writeheader(); w.writerows(tau_conditions)

# ## Step 3: Layer 1 — Aggregate tau
# Pool pairs across conditions before computing tau.


# Step 3: Layer 1 — Aggregate tau at all analysis levels

def aggregate_tau(conditions, group_keys, level_label):
    groups = defaultdict(lambda: {'c': 0, 'd': 0, 't': 0, 'n': 0})
    for tc in conditions:
        key = ('Overall',) if group_keys == ['Overall'] else tuple(tc[k] for k in group_keys)
        groups[key]['c'] += tc['Concordant']; groups[key]['d'] += tc['Discordant']
        groups[key]['t'] += tc['Tied']; groups[key]['n'] += 1
    results = []
    for key, v in sorted(groups.items()):
        total = v['c'] + v['d'] + v['t']
        tau = round((v['c'] - v['d']) / total, 4) if total > 0 else 0.0
        label = 'All conditions' if group_keys == ['Overall'] else ', '.join(f'{k}={val}' for k, val in zip(group_keys, key))
        results.append({'Agg_Level': level_label, 'Label': label, 'Tau': tau,
                        'Concordant': v['c'], 'Discordant': v['d'], 'Tied': v['t'],
                        'N_pairs': total, 'N_groups': v['n']})
    return results

all_agg = []
for keys, label in [
    (['Overall'], 'Overall'), (['LLM'], 'By LLM'), (['Dilemma'], 'By Dilemma'),
    (['Prompting'], 'By Prompting'), (['Domain'], 'By Domain'),
    (['LLM', 'Dilemma'], 'By LLM × Dilemma'), (['LLM', 'Prompting'], 'By LLM × Prompting'),
    (['LLM', 'Dilemma', 'Prompting'], 'By LLM × Dilemma × Prompting'),
    (['Dilemma', 'Prompting'], 'By Dilemma × Prompting'),
]:
    all_agg.extend(aggregate_tau(tau_conditions, keys, label))

with open(os.path.join(OUTPUT_DIR, 'tau_aggregated.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Agg_Level', 'Label', 'Tau', 'Concordant', 'Discordant', 'Tied', 'N_pairs', 'N_groups'])
    w.writeheader(); w.writerows(all_agg)

ot = [a for a in all_agg if a['Label'] == 'All conditions'][0]
print(f"  Overall: tau = {ot['Tau']} (C={ot['Concordant']}, D={ot['Discordant']}, T={ot['Tied']})")

# ######## Step 4: Layer 1 — Spearman's rho (supplementary)
# ######## Supplementary metric only. Pools across dilemma axes.


# Step 4: Layer 1 — Spearman's rho (supplementary)

spearman_results = []
for mf in [None] + MODEL_ORDER:
    filt = rows if mf is None else [r for r in rows if r['LLM_std'] == mf]
    label = 'Overall (all models)' if mf is None else mf
    means = defaultdict(list)
    for r in filt: means[(r['Language'], r['Dilemma'])].append(r['Fraction_A'])
    fa_m, mme_s = [], []
    for (lang, dil), fl in sorted(means.items()):
        fa_m.append(sum(fl)/len(fl))
        mme_s.append([r['MME_Score'] for r in filt if r['Language'] == lang and r['Dilemma'] == dil][0])
    rho = spearman_rho(fa_m, mme_s)
    spearman_results.append({'Model': label, 'Rho': rho, 'N_points': len(fa_m)})
    print(f"  {label}: rho = {rho}")

with open(os.path.join(OUTPUT_DIR, 'spearman_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Model', 'Rho', 'N_points'])
    w.writeheader(); w.writerows(spearman_results)

# ######## Step 5: Layer 2 — Three-dimensional typology
# ######## Categorize matrix of Differentiation, direction, deliberation into 7 bins.


# Step 5: Layer 2 — What type of governance failure does each cell exhibit?
#
# Three dimensions, each on its own scale:
#   1 (primary): Gradient fit — local concordance (EQ3-4)
#   2 (secondary): Directional alignment (EQ5)
#   3 (tertiary): Deliberation quality (EQ6-7)
# Combined into 7 bins (EQ8).

orig_fas = [r['Fraction_A'] for r in rows]
orig_mmes = [r['MME_Score'] for r in rows]
typology_results = compute_full_typology(orig_fas, orig_mmes, condition_groups)

for i, r in enumerate(rows):
    r.update(typology_results[i])

assert sum(1 for r in rows if r['Bin_Score'] == 0) == 0, "Unclassified cells"
bc = Counter(r['Bin_Score'] for r in rows)
for b in sorted(bc): print(f"  Bin {b} ({BIN_NAMES[b]}): {bc[b]}/576 = {100*bc[b]/576:.1f}%")

# ######## Step 6: Compile master summary
# ######## revised_master_v3.csv with all columns.


# Step 6: Consolidate into master summary

fields = [
    'Dilemma', 'Domain', 'Language', 'Country', 'Prompting', 'LLM', 'LLM_std', 'Scenario_ID',
    'MME_Score', 'Fraction_A', 'SD', 'CI_Lower', 'CI_Upper', 'P_Value', 'Is_Deterministic',
    'LC', 'Gradient_Fit', 'Direction', 'Deliberation', 'Del_Group', 'Cell_Type', 'Bin_Score'
]
with open(os.path.join(OUTPUT_DIR, 'revised_master_v3.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)

# ## Step 7: Layer 2 - Typology summaries
# 13 CSV files at every aggregation level.


# Step 7: Layer 2 - Typology summaries at all aggregation levels

BIN_ORDER = [1, 2, 3, 4, 5, 6, 7]

def compute_typology_summary(data_rows, group_keys):
    groups = defaultdict(lambda: Counter())
    totals = defaultdict(int)
    dets = defaultdict(int)
    for r in data_rows:
        key = ('Overall',) if group_keys == ['Overall'] else tuple(r[k] for k in group_keys)
        groups[key][r['Bin_Score']] += 1; totals[key] += 1
        if r['Is_Deterministic'] == 1: dets[key] += 1
    results = []
    for key in sorted(groups.keys()):
        n = totals[key]
        row = {}
        if group_keys != ['Overall']:
            for i, k in enumerate(group_keys): row[k] = key[i]
        else: row['Grouping'] = 'Overall'
        row['N'] = n
        for b in BIN_ORDER:
            nm = BIN_NAMES[b].replace(' ','_').replace('-','_')
            c = groups[key].get(b, 0)
            row[f'Bin{b}_{nm}_count'] = c; row[f'Bin{b}_{nm}_pct'] = round(100*c/n, 1)
        row['Tracking_A_bins12_count'] = groups[key].get(1,0) + groups[key].get(2,0)
        row['Tracking_A_bins12_pct'] = round(100*row['Tracking_A_bins12_count']/n, 1)
        row['All_tracking_bins123_count'] = row['Tracking_A_bins12_count'] + groups[key].get(3,0)
        row['All_tracking_bins123_pct'] = round(100*row['All_tracking_bins123_count']/n, 1)
        row['NonTracking_B_bins67_count'] = groups[key].get(6,0) + groups[key].get(7,0)
        row['NonTracking_B_bins67_pct'] = round(100*row['NonTracking_B_bins67_count']/n, 1)
        row['Deterministic_count'] = dets[key]
        row['Deterministic_pct'] = round(100*dets[key]/n, 1)
        results.append(row)
    return results

for keys, fn in [
    (['Overall'], 'typology_summary_overall.csv'),
    (['LLM_std'], 'typology_summary_by_model.csv'),
    (['Dilemma'], 'typology_summary_by_dilemma.csv'),
    (['Language'], 'typology_summary_by_language.csv'),
    (['Prompting'], 'typology_summary_by_prompting.csv'),
    (['Domain'], 'typology_summary_by_domain.csv'),
    (['LLM_std','Dilemma'], 'typology_summary_by_model_dilemma.csv'),
    (['LLM_std','Language'], 'typology_summary_by_model_language.csv'),
    (['LLM_std','Prompting'], 'typology_summary_by_model_prompting.csv'),
    (['Language','Dilemma'], 'typology_summary_by_language_dilemma.csv'),
    (['Dilemma','Prompting'], 'typology_summary_by_dilemma_prompting.csv'),
    (['LLM_std','Dilemma','Prompting'], 'typology_summary_by_model_dilemma_prompting.csv'),
    (['LLM_std','Language','Dilemma'], 'typology_summary_by_model_language_dilemma.csv'),
]:
    s = compute_typology_summary(rows, keys)
    with open(os.path.join(OUTPUT_DIR, fn), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(s[0].keys())); w.writeheader(); w.writerows(s)

# ######## Step 8: Positional bias test for Layer 1
# ######## Tau under FA shift. Invariant except at floor.


# Step 8: Positional bias stress test — Layer 1
#
# Subtract delta from all FA (clipped at 0). Recompute tau.
# Tests: does the headline finding survive if some A-preference is positional artifact?
# Tau is invariant to uniform shifts EXCEPT at the floor (cells near FA=0 get clipped to 0, creating ties).
# =================

for delta in POSBIAS_DELTAS:
    print(f"\n  delta = {delta}:")

    # Per-condition tau comparison
    tau_comparison = []
    total_co, total_do, total_to = 0, 0, 0
    total_ca, total_da, total_ta = 0, 0, 0
    ordering_reversals = 0
    new_ties = 0

    for key, indices in sorted(condition_groups.items()):
        group = sorted([rows[i] for i in indices], key=lambda x: x['Language'])
        orig_fa = [r['Fraction_A'] for r in group]
        adj_fa = [max(0.0, round(fa - delta, 4)) for fa in orig_fa]
        mmes = [r['MME_Score'] for r in group]
        langs = [r['Language'] for r in group]

        tau_o, co, do, to_ = kendall_tau(orig_fa, mmes)
        tau_a, ca, da, ta = kendall_tau(adj_fa, mmes)

        total_co += co; total_do += do; total_to += to_
        total_ca += ca; total_da += da; total_ta += ta

        # Check pair-level changes
        for i in range(4):
            for j in range(i+1, 4):
                o_sign = (orig_fa[i] > orig_fa[j]) - (orig_fa[i] < orig_fa[j])
                a_sign = (adj_fa[i] > adj_fa[j]) - (adj_fa[i] < adj_fa[j])
                if o_sign != 0 and a_sign == 0: new_ties += 1
                if o_sign != 0 and a_sign != 0 and o_sign != a_sign: ordering_reversals += 1

        tau_comparison.append({
            'LLM': LLM_NAME_MAP.get(key[0], key[0]), 'Dilemma': key[1],
            'Domain': key[2], 'Prompting': key[3],
            'Tau_original': tau_o, 'Tau_adjusted': tau_a,
            'Tau_change': round(tau_a - tau_o, 4),
            'Tau_changed': tau_o != tau_a,
            'FA_original': '|'.join(f"{l}={v}" for l, v in zip(langs, orig_fa)),
            'FA_adjusted': '|'.join(f"{l}={v}" for l, v in zip(langs, adj_fa)),
        })

    # Aggregated comparison
    tot_o = total_co + total_do + total_to
    tot_a = total_ca + total_da + total_ta
    tau_agg_orig = round((total_co - total_do) / tot_o, 4) if tot_o > 0 else 0
    tau_agg_adj = round((total_ca - total_da) / tot_a, 4) if tot_a > 0 else 0

    cond_changed = sum(1 for t in tau_comparison if t['Tau_changed'])

    print(f"    Overall tau: {tau_agg_orig} → {tau_agg_adj} (change: {tau_agg_adj - tau_agg_orig:+.4f})")
    print(f"    Conditions with tau change: {cond_changed}/144")
    print(f"    Ordering reversals: {ordering_reversals}/864 pairs")
    print(f"    New ties from floor clipping: {new_ties}/864 pairs")
    print(f"    Tied pairs: {total_to} → {total_ta} (+{total_ta - total_to})")

    # Per-model aggregated tau under shift
    agg_rows = [{'Label': 'Overall', 'Tau_original': tau_agg_orig, 'Tau_adjusted': tau_agg_adj,
                 'Change': round(tau_agg_adj - tau_agg_orig, 4), 'Ordering_reversals': ordering_reversals,
                 'New_ties': new_ties}]

    for model in MODEL_ORDER:
        mc = [t for t in tau_comparison if t['LLM'] == model]
        # Recompute from per-condition
        m_orig_cdt = [(tc['Concordant'], tc['Discordant'], tc['Tied'])
                      for tc in tau_conditions if tc['LLM'] == model]
        m_co_sum = sum(x[0] for x in m_orig_cdt)
        m_do_sum = sum(x[1] for x in m_orig_cdt)
        m_to_sum = sum(x[2] for x in m_orig_cdt)
        m_tot_o = m_co_sum + m_do_sum + m_to_sum
        m_tau_o = round((m_co_sum - m_do_sum) / m_tot_o, 4) if m_tot_o > 0 else 0

        # Adjusted: recompute per condition
        m_ca_sum, m_da_sum, m_ta_sum = 0, 0, 0
        for key, indices in condition_groups.items():
            if LLM_NAME_MAP.get(key[0], key[0]) != model: continue
            group = sorted([rows[i] for i in indices], key=lambda x: x['Language'])
            adj_fa = [max(0.0, round(r['Fraction_A'] - delta, 4)) for r in group]
            mmes = [r['MME_Score'] for r in group]
            _, ca, da, ta = kendall_tau(adj_fa, mmes)
            m_ca_sum += ca; m_da_sum += da; m_ta_sum += ta
        m_tot_a = m_ca_sum + m_da_sum + m_ta_sum
        m_tau_a = round((m_ca_sum - m_da_sum) / m_tot_a, 4) if m_tot_a > 0 else 0

        agg_rows.append({'Label': model, 'Tau_original': m_tau_o, 'Tau_adjusted': m_tau_a,
                         'Change': round(m_tau_a - m_tau_o, 4), 'Ordering_reversals': '', 'New_ties': ''})

    # Write per-condition file
    with open(os.path.join(OUTPUT_DIR, f'posbias_layer1_delta_{delta}_per_condition.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tau_comparison[0].keys()))
        w.writeheader(); w.writerows(tau_comparison)

    # Write aggregated summary
    with open(os.path.join(OUTPUT_DIR, f'posbias_layer1_delta_{delta}_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)

    print(f"    Written: posbias_layer1_delta_{delta}_per_condition.csv, posbias_layer1_delta_{delta}_summary.csv")

# ######## Step 9: Positional bias test for Layer 2


# Step 9: Positional bias stress test for Layer 2
#
# Subtract delta from all FA (clipped at 0). Recompute full typology.
# Key question: how many cells change governance bin?
# Direction (0.50 threshold) is most sensitive to uniform shift.

for delta in POSBIAS_DELTAS:
    print(f"\n  delta = {delta}:")

    adj_fas = [max(0.0, round(r['Fraction_A'] - delta, 4)) for r in rows]
    adj_mmes = [r['MME_Score'] for r in rows]
    adj_typology = compute_full_typology(adj_fas, adj_mmes, condition_groups)

    # Compare bins
    cell_changes = []
    bin_change_count = 0
    direction_flips = 0
    gradient_fit_changes = 0

    for i in range(len(rows)):
        orig = rows[i]
        adj = adj_typology[i]
        changed = orig['Bin_Score'] != adj['Bin_Score']
        if changed: bin_change_count += 1
        if orig['Direction'] != adj['Direction']: direction_flips += 1
        if orig['Gradient_Fit'] != adj['Gradient_Fit']: gradient_fit_changes += 1

        cell_changes.append({
            'LLM_std': orig['LLM_std'], 'Dilemma': orig['Dilemma'], 'Domain': orig['Domain'],
            'Prompting': orig['Prompting'], 'Language': orig['Language'],
            'FA_original': orig['Fraction_A'], 'FA_adjusted': adj_fas[i],
            'LC_original': orig['LC'], 'LC_adjusted': adj['LC'],
            'Bin_original': orig['Bin_Score'], 'Bin_adjusted': adj['Bin_Score'],
            'Type_original': orig['Cell_Type'], 'Type_adjusted': adj['Cell_Type'],
            'Bin_changed': changed,
            'GradientFit_original': orig['Gradient_Fit'], 'GradientFit_adjusted': adj['Gradient_Fit'],
            'Direction_original': orig['Direction'], 'Direction_adjusted': adj['Direction'],
            'Deliberation_original': orig['Deliberation'], 'Deliberation_adjusted': adj_typology[i]['Deliberation'],
            'Floor_clipped': float(orig['Fraction_A']) <= delta,
        })

    # Bin distribution comparison
    orig_bins = Counter(r['Bin_Score'] for r in rows)
    adj_bins = Counter(r['Bin_Score'] for r in adj_typology)

    floor_clipped = sum(1 for c in cell_changes if c['Bin_changed'] and c['Floor_clipped'])

    print(f"    Cells with bin change: {bin_change_count}/576 ({100*bin_change_count/576:.1f}%)")
    print(f"    Direction flips (A→B): {direction_flips}/576")
    print(f"    Gradient fit changes: {gradient_fit_changes}/576")
    print(f"    Floor-clipped edge cases: {floor_clipped}/{bin_change_count}")
    print(f"    Bin distribution shift:")
    for b in BIN_ORDER:
        o = orig_bins.get(b, 0); a = adj_bins.get(b, 0)
        if o != a: print(f"      Bin {b} ({BIN_NAMES[b]}): {o} → {a} ({a-o:+d})")

    # Write per-cell changes (only changed cells for readability)
    changed_cells = [c for c in cell_changes if c['Bin_changed']]
    with open(os.path.join(OUTPUT_DIR, f'posbias_layer2_delta_{delta}_changed_cells.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cell_changes[0].keys()))
        w.writeheader(); w.writerows(changed_cells)

    # Write summary
    summary = [{'Metric': 'Cells with bin change', 'Value': bin_change_count, 'Of_total': 576},
               {'Metric': 'Direction flips', 'Value': direction_flips, 'Of_total': 576},
               {'Metric': 'Gradient fit changes', 'Value': gradient_fit_changes, 'Of_total': 576},
               {'Metric': 'Floor-clipped edge cases', 'Value': floor_clipped, 'Of_total': bin_change_count}]
    for b in BIN_ORDER:
        summary.append({'Metric': f'Bin {b} ({BIN_NAMES[b]})',
                        'Value': f"{orig_bins.get(b,0)} → {adj_bins.get(b,0)} ({adj_bins.get(b,0)-orig_bins.get(b,0):+d})",
                        'Of_total': 576})

    with open(os.path.join(OUTPUT_DIR, f'posbias_layer2_delta_{delta}_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['Metric', 'Value', 'Of_total'])
        w.writeheader(); w.writerows(summary)

    print(f"    Written: posbias_layer2_delta_{delta}_changed_cells.csv, posbias_layer2_delta_{delta}_summary.csv")

# ######## Step 10: Reproducibility check
# ######## Confirms all headline values reported in the paper


# Step 10: Reproducibility check

rpt = []; pc = fc = 0
def V(cond, desc, exp, act):
    global pc, fc
    if cond: pc += 1; rpt.append(f"  ✓ {desc}: {act}")
    else: fc += 1; rpt.append(f"  ✗ FAIL {desc}: expected={exp}, actual={act}")

rpt.append("=" * 80)
rpt.append("Verification report — analysis pipeline v3")
rpt.append("=" * 80)

rpt.append("\n--- Layer 1 ---\n")
V(abs(ot['Tau']-0.0856)<0.001, "Overall tau", 0.0856, ot['Tau'])
V(ot['Concordant']==311, "C", 311, ot['Concordant'])
V(ot['Discordant']==237, "D", 237, ot['Discordant'])
V(ot['Tied']==316, "T", 316, ot['Tied'])
for m, et in [('GPT-4o',0.0093),('DeepSeek R1',-0.0509),('Mistral Large',0.3148),('Gemini Flash',0.0694)]:
    V(abs([a for a in all_agg if a['Label']==f'LLM={m}'][0]['Tau']-et)<0.001, f"{m} tau", et, [a for a in all_agg if a['Label']==f'LLM={m}'][0]['Tau'])
for d, et in [('MF',0.184),('YO',0.0694),('HL',0.0035)]:
    V(abs([a for a in all_agg if a['Label']==f'Dilemma={d}'][0]['Tau']-et)<0.001, f"{d} tau", et, [a for a in all_agg if a['Label']==f'Dilemma={d}'][0]['Tau'])
for p, et in [('PLAIN',0.0509),('CC',0.0509),('ZSCOT',-0.0787),('FSCOT',0.3194)]:
    V(abs([a for a in all_agg if a['Label']==f'Prompting={p}'][0]['Tau']-et)<0.001, f"{p} tau", et, [a for a in all_agg if a['Label']==f'Prompting={p}'][0]['Tau'])
for d, et in [('D1',0.059),('D2',0.1042),('D3',0.0938)]:
    V(abs([a for a in all_agg if a['Label']==f'Domain={d}'][0]['Tau']-et)<0.001, f"{d} tau", et, [a for a in all_agg if a['Label']==f'Domain={d}'][0]['Tau'])
for label, et in [('LLM=Mistral Large, Dilemma=HL, Prompting=FSCOT', 0.833),
                   ('LLM=DeepSeek R1, Dilemma=HL, Prompting=ZSCOT', -0.611)]:
    matches = [a for a in all_agg if a['Label'] == label]
    if matches: V(abs(matches[0]['Tau'] - et) < 0.01, f"Condition {label}", et, matches[0]['Tau'])

rpt.append("\n--- Layer 2 ---\n")
for b, ec in [(1,48),(2,94),(3,89),(4,163),(5,113),(6,33),(7,36)]:
    V(bc[b]==ec, f"Bin {b} ({BIN_NAMES[b]})", ec, bc[b])
V(sum(bc.values())==576, "Total", 576, sum(bc.values()))

b4 = [r for r in rows if r['Bin_Score']==4]
V(sum(1 for r in b4 if r['Del_Group']=='Det/Near-det')==163, "Bin 4 all Det/Near-det", 163, sum(1 for r in b4 if r['Del_Group']=='Det/Near-det'))
V(sum(1 for r in b4 if r['Fraction_A']==1.0)==143, "Bin 4 FA=1.00", 143, sum(1 for r in b4 if r['Fraction_A']==1.0))
V(sum(1 for r in b4 if r['Fraction_A']==0.0)==14, "Bin 4 FA=0.00", 14, sum(1 for r in b4 if r['Fraction_A']==0.0))

b3 = [r for r in rows if r['Bin_Score']==3]
V(sum(1 for r in b3 if r['Del_Group']=='Variable')==51, "Bin 3 Variable", 51, sum(1 for r in b3 if r['Del_Group']=='Variable'))
V(sum(1 for r in b3 if r['LLM_std']=='Mistral Large')==48, "Bin 3 Mistral", 48, sum(1 for r in b3 if r['LLM_std']=='Mistral Large'))
V(sum(1 for r in rows if r['Is_Deterministic']==1)==302, "Deterministic cells", 302, sum(1 for r in rows if r['Is_Deterministic']==1))

rpt.append(f"\n  {pc} passed, {fc} failed")

rt = '\n'.join(rpt)
with open(os.path.join(OUTPUT_DIR, 'VERIFICATION_REPORT.txt'), 'w') as f: f.write(rt)

# ######## Step 11: Package outputs
# ######## ZIP all files.


# Step 11: Package all outputs

with zipfile.ZipFile('analysis_v3_outputs.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in sorted(os.listdir(OUTPUT_DIR)): zf.write(os.path.join(OUTPUT_DIR, fn), fn)
fc_files = len(os.listdir(OUTPUT_DIR))
print(f"  analysis_v3_outputs.zip ({fc_files} files)")

print("\n" + rt)
print(f"\nDone. {fc_files} files in {OUTPUT_DIR}/")

# ## Download outputs
