# -*- coding: utf-8 -*-

# =============================================================================
# Figure 6: Governance typology heatmap (576 cells)
# =============================================================================
#
# Input:  revised_master_v3.csv (from 05_Analysis/)
# Output: figure_6.png (seven-bin typology heatmap)
#
# Colour scheme:
#   Bin 1 (Calibrated):                    #4169E1 (royal blue)
#   Bin 2 (Rigid tracking):                 #87CEEB (sky blue)
#   Bin 3 (Gradient-sensitive overshoot):   #B0E0E6 (powder blue)
#   Bin 4 (Gradient erased):                hatched (#F0F8FF + #87CEEB stripes)
#   Bin 5 (Gradient inverted):              #FFB6C1 (light pink)
#   Bin 6 (Non-tracking contradiction):     #FF69B4 (hot pink)
#   Bin 7 (Non-tracking rigidity):          #FF0000 (red)
#
# Dependencies: pandas, matplotlib, numpy
#
# Originally run in Google Colab. For standalone use, replace
# upload_file() call with pd.read_csv('path/to/revised_master_v3.csv').

##### Figure 6: Layer 2 Governance typology heatmap
##### File needed: revised_master_v3.csv


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json, re, io
from collections import Counter
from google.colab import files

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})

# Upload
def upload_file(prompt_name):
    print(f"Upload {prompt_name}:")
    uploaded = files.upload()
    base = prompt_name.replace('.csv', '').replace('.json', '')
    for fname, content in uploaded.items():
        clean = re.sub(r'\s*\(\d+\)', '', fname.replace('.csv', '').replace('.json', ''))
        if base.lower() in clean.lower():
            return pd.read_csv(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

# Load data from CSV
df = upload_file('revised_master_v3.csv')
print(f"Loaded {len(df)} rows from CSV")

# Build lookup: key = "LLM_std|Prompting|Dilemma|Domain|Language" → Bin_Score
hm_data = {}
for _, row in df.iterrows():
    key = f"{row['LLM_std']}|{row['Prompting']}|{row['Dilemma']}|{row['Domain']}|{row['Language']}"
    hm_data[key] = int(row['Bin_Score'])

print(f"Built {len(hm_data)} cell lookups")

# Constants
MODEL_KEYS   = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini Flash']
MODEL_LABELS = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini 2.0\nFlash Lite']
PROMPT_ORDER  = ['PLAIN', 'CC', 'ZSCOT', 'FSCOT']
PROMPT_LABELS = ['Plain', 'Cultural Calibration', 'Zero-Shot CoT', 'Few-Shot CoT']
DILEMMA_ORDER = ['MF', 'YO', 'HL']
DOMAIN_ORDER  = ['D1', 'D2', 'D3']
LANG_ORDER    = ['EN', 'CN', 'JP', 'ES']
LANG_COLORS   = {'EN': '#4E79A7', 'CN': '#E15759', 'JP': '#59A14F', 'ES': '#EDC948'}

BIN_COLORS = {
    1: '#4169E1', 2: '#87CEEB', 3: '#B0E0E6', 4: '#F0F8FF',
    5: '#FFB6C1', 6: '#FF69B4', 7: '#FF0000',
}
BIN_NAMES = {
    1: 'Calibrated', 2: 'Rigid tracking', 3: 'Gradient-sensitive overshoot',
    4: 'Gradient erased', 5: 'Gradient inverted',
    6: 'Non-tracking contradiction', 7: 'Non-tracking rigidity',
}
BIN_TEXT_COLOR = {
    1: '#FFFFFF', 2: '#1a1a1a', 3: '#1a1a1a', 4: '#1a1a1a',
    5: '#1a1a1a', 6: '#1a1a1a', 7: '#FFFFFF',
}

bin_counts = Counter(hm_data.values())

# Layout
CELL_SIZE = 0.38
panel_w = 9 * CELL_SIZE   # 3.42"
panel_h = 4 * CELL_SIZE   # 1.52"

LEFT_MARGIN = 2.0
H_GAP = 0.28
V_GAP = 0.30
TOP_HEADER = 4.15
BOTTOM_PAD = 0.15

fig_w = LEFT_MARGIN + 4 * panel_w + 3 * H_GAP + 0.2
fig_h = TOP_HEADER + 4 * panel_h + 3 * V_GAP + BOTTOM_PAD

print(f"Figure size: {fig_w:.1f} x {fig_h:.1f} inches")

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor='white')

def y_from_top(inches_from_top):
    return 1.0 - inches_from_top / fig_h
def x_frac(inches):
    return inches / fig_w


# TITLE — 24pt bold

fig.text(x_frac(0.3), y_from_top(0.35), 'Figure 6: Governance typology heatmap',
         ha='left', va='top', fontsize=24, fontweight='bold', color='#1a1a1a')


# SUBTITLE — 17pt

fig.text(x_frac(0.3), y_from_top(0.85),
         'Mistral Large is the only predominantly blue model; GPT-4o shows highest rates of gradient erasure.',
         ha='left', va='top', fontsize=17, color='#222222')
fig.text(x_frac(0.3), y_from_top(1.12),
         'Across models, few-shot chain-of-thought shifts patterns toward stronger gradient tracking (more blue), albeit unevenly.',
         ha='left', va='top', fontsize=17, color='#222222')
fig.text(x_frac(0.3), y_from_top(1.39),
         'Within each panel: rows = country-language pair; columns = dilemma × domain.',
         ha='left', va='top', fontsize=17, color='#222222')


# LEGEND — 2 rows, 16pt text

swatch_h_in = 0.28
swatch_w_in = 0.32
legend_font = 16

def draw_legend_item(b, x_in, y_top):
    sx = x_frac(x_in)
    sy = y_from_top(y_top + swatch_h_in)
    sw = swatch_w_in / fig_w
    sh = swatch_h_in / fig_h
    fig.patches.append(mpatches.FancyBboxPatch(
        (sx, sy), sw, sh, boxstyle="round,pad=0.002",
        facecolor=BIN_COLORS[b], edgecolor='#888888', linewidth=0.8,
        transform=fig.transFigure, clip_on=False
    ))
    if b == 4:
        fig.patches.append(mpatches.FancyBboxPatch(
            (sx, sy), sw, sh, boxstyle="round,pad=0.002",
            facecolor='none', edgecolor='#d4a843',
            linewidth=0, hatch='///', alpha=0.70,
            transform=fig.transFigure, clip_on=False
        ))
    label = f"{b}: {BIN_NAMES[b]}"
    text_x = x_frac(x_in + swatch_w_in + 0.10)
    fig.text(text_x, y_from_top(y_top + swatch_h_in / 2), label,
             ha='left', va='center', fontsize=legend_font, fontweight='bold', color='#1a1a1a')

row1_y = 1.85
row2_y = row1_y + swatch_h_in + 0.14

draw_legend_item(1, 0.3, row1_y)
draw_legend_item(2, 2.7, row1_y)
draw_legend_item(3, 5.6, row1_y)
draw_legend_item(4, 10.2, row1_y)

draw_legend_item(5, 0.3, row2_y)
draw_legend_item(6, 3.5, row2_y)
draw_legend_item(7, 8.0, row2_y)


# PROMPTING LABELS — 19pt and bold

prompt_y_top = 3.30
for pi in range(4):
    px_in = LEFT_MARGIN + pi * (panel_w + H_GAP)
    center_x = x_frac(px_in + panel_w / 2)
    fig.text(center_x, y_from_top(prompt_y_top),
             PROMPT_LABELS[pi], ha='center', va='bottom',
             fontsize=19, fontweight='bold', color='#1a1a1a')


# DILEMMA HEADERS — 15pt ; DOMAIN — 9pt dark grey
dilemma_y_top = 3.70
domain_y_top = 3.98

for pi in range(4):
    px_in = LEFT_MARGIN + pi * (panel_w + H_GAP)
    for di, dilemma in enumerate(DILEMMA_ORDER):
        center_x = x_frac(px_in + di * 3 * CELL_SIZE + 1.5 * CELL_SIZE)
        fig.text(center_x, y_from_top(dilemma_y_top), dilemma,
                 ha='center', va='bottom', fontsize=15, fontweight='bold', color='#1a1a1a')
    for di in range(3):
        for ddi, domain in enumerate(DOMAIN_ORDER):
            col = di * 3 + ddi
            cx = x_frac(px_in + col * CELL_SIZE + CELL_SIZE / 2)
            fig.text(cx, y_from_top(domain_y_top), domain,
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#374151')

# DATA PANELS

data_y_top = TOP_HEADER

for mi, model_key in enumerate(MODEL_KEYS):
    for pi, prompt in enumerate(PROMPT_ORDER):
        px_in = LEFT_MARGIN + pi * (panel_w + H_GAP)
        py_top_in = data_y_top + mi * (panel_h + V_GAP)

        left = px_in / fig_w
        bottom = 1.0 - (py_top_in + panel_h) / fig_h
        width = panel_w / fig_w
        height = panel_h / fig_h

        ax = fig.add_axes([left, bottom, width, height])
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 4)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for li, lang in enumerate(LANG_ORDER):
            for di, dilemma in enumerate(DILEMMA_ORDER):
                for ddi, domain in enumerate(DOMAIN_ORDER):
                    col = di * 3 + ddi
                    key = f"{model_key}|{prompt}|{dilemma}|{domain}|{lang}"
                    score = hm_data.get(key)
                    if score is None:
                        continue

                    ax.add_patch(mpatches.FancyBboxPatch(
                        (col + 0.04, li + 0.04), 0.92, 0.92,
                        boxstyle="round,pad=0.02",
                        facecolor=BIN_COLORS[score], edgecolor='#d1d5db', linewidth=0.5
                    ))
                    if score == 4:
                        ax.add_patch(mpatches.FancyBboxPatch(
                            (col + 0.04, li + 0.04), 0.92, 0.92,
                            boxstyle="round,pad=0.02",
                            facecolor='none', edgecolor='#d4a843',
                            linewidth=0, hatch='///', alpha=0.70
                        ))
                    ax.text(col + 0.5, li + 0.5, str(score),
                            ha='center', va='center',
                            fontsize=9, fontweight='bold', color=BIN_TEXT_COLOR[score])

        for sep_x in [3, 6]:
            ax.axvline(x=sep_x, color='#aaaaaa', linewidth=0.7)

        if pi == 0:
            for li, lang in enumerate(LANG_ORDER):
                ax.text(-0.15, li + 0.5, lang, ha='right', va='center',
                        fontsize=12, fontweight='bold', color=LANG_COLORS[lang])

    # MODEL LABEL — 16pt bold
    model_center_y_in = data_y_top + mi * (panel_h + V_GAP) + panel_h / 2
    fig.text(x_frac(0.65), y_from_top(model_center_y_in),
             MODEL_LABELS[mi], ha='center', va='center',
             fontsize=16, fontweight='bold', color='#1a1a1a', linespacing=1.3)

plt.savefig('figure_6.png', dpi=300, bbox_inches='tight',
            pad_inches=0.12, facecolor='white')
plt.show()
print("Saved: figure_6.png")

