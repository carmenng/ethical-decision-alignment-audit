# -*- coding: utf-8 -*-

# =============================================================================
# Figure 3, 4, 5: Layer 1 gradient concordance visualisations
# =============================================================================
#
# Input:  tau_aggregated.csv (from 05_Analysis/)
#         spearman_results.csv (from 05_Analysis/)
# Output: figure_3.png (tau by model with rho and tied-pair rates)
#         figure_45.png (tau by model x prompting, tau by model x dilemma)
#         figure_345_combined.png (combined layout)
#
# Dependencies: pandas, matplotlib, numpy, PIL
#
# Originally run in Google Colab. For standalone use, replace
# upload_csv() calls with pd.read_csv('path/to/file.csv').

# Figure 3: Per-model concordance bar chart
# Files needed: `tau_aggregated.csv` and `spearman_results.csv`


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io, re
from google.colab import files

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})

def upload_csv(prompt_name):
    """Upload a CSV, handling Colab's duplicate naming like 'file (3).csv'."""
    print(f"Upload {prompt_name}:")
    uploaded = files.upload()
    # Find the file matching the base name (ignoring ' (N)' duplicates)
    base = prompt_name.replace('.csv', '')
    for fname, content in uploaded.items():
        clean = re.sub(r'\s*\(\d+\)', '', fname.replace('.csv', ''))
        if base.lower() in clean.lower():
            return pd.read_csv(io.BytesIO(content))
    # Fallback: just use whatever was uploaded
    return pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

##### Upload files ##### 
tau_df = upload_csv('tau_aggregated.csv')
spearman_df = upload_csv('spearman_results.csv')

##### Extract data ##### 
model_order = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini Flash']

BAR_TAU = []
BAR_TIE = []
model_tau = tau_df[tau_df['Agg_Level'] == 'By LLM']
for m in model_order:
    row = model_tau[model_tau['Label'].str.contains(m)]
    tau_val = round(row['Tau'].values[0], 2)
    tied = row['Tied'].values[0]
    total = row['Concordant'].values[0] + row['Discordant'].values[0] + tied
    BAR_TAU.append(tau_val)
    BAR_TIE.append(round(tied / total * 100, 1))

BAR_RHO = []
rho_col = [c for c in spearman_df.columns if c.lower() == 'rho'][0]
label_col = [c for c in spearman_df.columns if c.lower() in ('label', 'model')][0]
for m in model_order:
    row = spearman_df[spearman_df[label_col].str.contains(m)]
    BAR_RHO.append(round(row[rho_col].values[0], 2))

print(f"\u03C4: {BAR_TAU}")
print(f"\u03C1: {BAR_RHO}")
print(f"Tie: {BAR_TIE}")

##### Chart ##### 
PURPLE = '#7c3aed'
NEAR_BLACK = '#1a1a1a'

def bc(v):
    if v > 0.08: return '#60a5fa'
    if v < -0.08: return '#f87171'
    return '#d1d5db'
def bs(v):
    if v > 0.08: return '#3b82f6'
    if v < -0.08: return '#ef4444'
    return '#9ca3af'

fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
ax.set_position([0.125, 0.11, 0.775, 0.68])
ax.set_ylim(-0.18, 0.60); ax.set_xlim(-0.5, 3.8)

for v in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    ax.axhline(v, color='#9ca3af' if v==0 else '#f1f5f9', linewidth=0.8 if v==0 else 0.4, zorder=0)
ax.set_yticks([-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax.set_yticklabels([f'{v:+.1f}' for v in [-0.1,0,0.1,0.2,0.3,0.4,0.5]], fontsize=16, color='#555555', fontweight='600')

bw=0.32; mp=[0,1,2,3]
tx=[m-bw/2-0.02 for m in mp]; rx=[m+bw/2+0.02 for m in mp]
for i in range(4):
    ax.bar(tx[i], BAR_TAU[i], width=bw, color=bc(BAR_TAU[i]), edgecolor='none', zorder=2)
    ax.bar(rx[i], BAR_RHO[i], width=bw, facecolor='none', edgecolor=bs(BAR_RHO[i]), linewidth=2.0, linestyle=(0,(4,2)), zorder=2)

# GPT-4o
ax.text(tx[0], BAR_TAU[0]+0.02, f'{BAR_TAU[0]:+.2f}', ha='center', va='bottom', fontsize=18, fontweight='bold', color=NEAR_BLACK)
ax.text(rx[0], BAR_RHO[0]-0.02, f'{BAR_RHO[0]:+.2f}', ha='center', va='top', fontsize=13, color=NEAR_BLACK)
# DeepSeek
ax.text(tx[1]-0.08, -0.08, f'{BAR_TAU[1]:+.2f}', ha='center', va='top', fontsize=18, fontweight='bold', color=NEAR_BLACK)
ax.text(rx[1], BAR_RHO[1]-0.02, f'{BAR_RHO[1]:+.2f}', ha='center', va='top', fontsize=13, color=NEAR_BLACK)
# Mistral
ax.text(tx[2]-0.05, BAR_TAU[2]+0.03, f'{BAR_TAU[2]:+.2f}', ha='center', va='bottom', fontsize=18, fontweight='bold', color='#2563eb')
ax.text(rx[2], BAR_RHO[2]+0.02, f'{BAR_RHO[2]:+.2f}', ha='center', va='bottom', fontsize=13, color=NEAR_BLACK)
# Gemini
ax.text(tx[3], BAR_TAU[3]+0.02, f'{BAR_TAU[3]:+.2f}', ha='center', va='bottom', fontsize=18, fontweight='bold', color=NEAR_BLACK)
ax.text(rx[3]+bw/2+0.03, BAR_RHO[3], f'{BAR_RHO[3]:+.2f}', ha='left', va='center', fontsize=13, color=NEAR_BLACK)

# TIE RATE — PURPLE
axt=ax.twinx(); axt.set_ylim(0,70); axt.set_yticks([])
axt.plot(mp, BAR_TIE, color=PURPLE, linewidth=1.2, zorder=3, alpha=0.6)
for i in range(4):
    axt.scatter(mp[i], BAR_TIE[i], s=140, color=PURPLE, edgecolors='white', linewidths=1.5, zorder=4, alpha=0.8)
    pass  # labels placed individually below
# Tie rate labels — hand-placed
axt.text(mp[0]+0.08, BAR_TIE[0], f'{BAR_TIE[0]}%', fontsize=17, color=PURPLE, va='center', fontweight='600')
axt.text(mp[1]-0.25, BAR_TIE[1]-3, f'{BAR_TIE[1]}%', fontsize=17, color=PURPLE, va='center', ha='right', fontweight='600')
axt.text(mp[2]+0.08, BAR_TIE[2]-3, f'{BAR_TIE[2]}%', fontsize=17, color=PURPLE, va='top', fontweight='600')
axt.text(mp[3]+0.08, BAR_TIE[3], f'{BAR_TIE[3]}%', fontsize=17, color=PURPLE, va='center', fontweight='600')
for sp in axt.spines.values(): sp.set_visible(False)

ax.set_xticks(mp)
ax.set_xticklabels(['GPT-4o','DeepSeek\nR1','Mistral\nLarge','Gemini 2.0\nFlash Lite'], fontsize=15, color='#374151', fontweight='600', linespacing=1.15)
ax.tick_params(axis='x', length=0, pad=10); ax.tick_params(axis='y', length=0, pad=8)
for sp in ax.spines.values(): sp.set_visible(False)

fig.text(0.50, 0.965, 'Figure 3', ha='center', fontsize=24, fontweight='bold', color=NEAR_BLACK)
fig.text(0.50, 0.910, 'Only Mistral tracks pluralistic cultural', ha='center', fontsize=19, fontweight='bold', color='#222222')
fig.text(0.50, 0.877, 'preferences in both order and proportion', ha='center', fontsize=19, fontweight='bold', color='#222222')
fig.text(0.50, 0.845, 'High tie rate = identical output to different countries', ha='center', fontsize=16, fontweight='500', color='#333333')

fig.text(0.13, 0.008, '\u25A0', fontsize=18, color='#60a5fa')
fig.text(0.16, 0.008, '\u03C4 (ordering)', fontsize=15, color='#555555', va='center', fontweight='600')
fig.text(0.38, 0.008, '\u229F', fontsize=16, color='#3b82f6')
fig.text(0.41, 0.008, '\u03C1 (proportional)', fontsize=15, color='#555555', va='center', fontweight='600')
fig.text(0.68, 0.008, '\u25CF', fontsize=17, color=PURPLE)
fig.text(0.71, 0.008, 'Tie rate', fontsize=15, color=PURPLE, va='center', fontweight='600')

plt.savefig('figure_3.png', dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.show()
print("Saved: figure_3.png")

#####  Figures 4 & 5: Layer 1 concordance heatmaps
#####  File needed: tau_aggregated.csv


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import io, re
from google.colab import files

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})

def upload_csv(prompt_name):
    print(f"Upload {prompt_name}:")
    uploaded = files.upload()
    base = prompt_name.replace('.csv', '')
    for fname, content in uploaded.items():
        clean = re.sub(r'\s*\(\d+\)', '', fname.replace('.csv', ''))
        if base.lower() in clean.lower():
            return pd.read_csv(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

tau_df = upload_csv('tau_aggregated.csv')

model_order = ['GPT-4o', 'DeepSeek R1', 'Mistral Large', 'Gemini Flash']
prompt_order = ['PLAIN', 'CC', 'ZSCOT', 'FSCOT']
dilemma_order = ['MF', 'YO', 'HL']

mp_rows = tau_df[tau_df['Agg_Level'] == 'By LLM × Prompting']
TAU_MP = np.zeros((4, 4))
for mi, m in enumerate(model_order):
    for pi, p in enumerate(prompt_order):
        label = f'LLM={m}, Prompting={p}'
        row = mp_rows[mp_rows['Label'] == label]
        TAU_MP[mi, pi] = round(row['Tau'].values[0], 2)

md_rows = tau_df[tau_df['Agg_Level'] == 'By LLM × Dilemma']
TAU_MD = np.zeros((4, 3))
for mi, m in enumerate(model_order):
    for di, d in enumerate(dilemma_order):
        label = f'LLM={m}, Dilemma={d}'
        row = md_rows[md_rows['Label'] == label]
        TAU_MD[mi, di] = round(row['Tau'].values[0], 2)

print("Figure 4 (Model x Prompting):")
print(TAU_MP)
print("\nFigure 5 (Model x Dilemma):")
print(TAU_MD)

DZ = 0.10; MR = 0.60
def tau_bg(t):
    if abs(t) <= DZ: return '#ffffff'
    s = min(1.0, (abs(t) - DZ) / (MR - DZ))
    if t > 0: return f'#{int(255-s*195):02x}{int(255-s*115):02x}ff'
    return f'#ff{int(255-s*155):02x}{int(255-s*195):02x}'
def tau_text(t):
    if abs(t) <= DZ: return '#7a7067'
    if abs(t) > 0.35: return '#ffffff'
    return '#1e293b'

NEAR_BLACK = '#1a1a1a'

def draw_hm(ax, data, row_labels, col_labels, show_ylabel=True):
    nr, nc = data.shape
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.invert_yaxis()
    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            ax.add_patch(patches.FancyBboxPatch((j+0.06, i+0.06), 0.88, 0.88,
                boxstyle="round,pad=0.02", facecolor=tau_bg(v), edgecolor='#d1d5db', linewidth=0.5))
            if abs(v) <= DZ:
                ax.add_patch(patches.FancyBboxPatch((j+0.06, i+0.06), 0.88, 0.88,
                    boxstyle="round,pad=0.02", facecolor='none',
                    edgecolor='#8a7a6a', linewidth=0, hatch='///', alpha=0.40))
            s = '+' if v >= 0 else ''
            ax.text(j+0.5, i+0.5, f'{s}{v:.2f}', ha='center', va='center',
                    fontsize=22, fontweight='bold', color=tau_text(v))
    ax.set_xticks([x+0.5 for x in range(nc)])
    ax.set_xticklabels(col_labels, fontsize=20, fontweight='700', color='#374151')
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks([y+0.5 for y in range(nr)])
    if show_ylabel:
        ax.set_yticklabels(row_labels, fontsize=18, fontweight='700', color='#374151', linespacing=1.1)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0, pad=8)
    for sp in ax.spines.values(): sp.set_visible(False)

fig = plt.figure(figsize=(12, 9), dpi=300)

fig.text(0.35, 0.965, 'Figure 4', ha='center', fontsize=24, fontweight='bold', color=NEAR_BLACK)
fig.text(0.35, 0.905, 'Does prompting help LLMs rank', ha='center', fontsize=19, fontweight='bold', color='#222222')
fig.text(0.35, 0.875, "countries' preference strengths?", ha='center', fontsize=19, fontweight='bold', color='#222222')

fig.text(0.82, 0.965, 'Figure 5', ha='center', fontsize=24, fontweight='bold', color=NEAR_BLACK)
fig.text(0.82, 0.910, 'On which moral trade-offs do', ha='center', fontsize=19, fontweight='bold', color='#222222')
fig.text(0.82, 0.880, 'LLMs track cross-country', ha='center', fontsize=19, fontweight='bold', color='#222222')
fig.text(0.82, 0.850, 'preference gradients?', ha='center', fontsize=19, fontweight='bold', color='#222222')

ax4 = fig.add_axes([0.13, 0.20, 0.48, 0.56])
draw_hm(ax4, TAU_MP,
        ['GPT-4o', 'DeepSeek\nR1', 'Mistral\nLarge', 'Gemini 2.0\nFlash Lite'],
        ['PLAIN', 'CC', 'ZSCOT', 'FSCOT'])

ax5 = fig.add_axes([0.63, 0.20, 0.36, 0.56])
draw_hm(ax5, TAU_MD, ['']*4, ['MF', 'YO', 'HL'], show_ylabel=False)

cbar_ax = fig.add_axes([0.02, 0.12, 0.96, 0.025])
n = 120; gradient = np.zeros((1, n, 3))
for ci in range(n):
    t = -MR + (2*MR*ci)/(n-1); hx = tau_bg(t)
    gradient[0, ci] = [int(hx[1:3],16)/255, int(hx[3:5],16)/255, int(hx[5:7],16)/255]
cbar_ax.imshow(gradient, aspect='auto', extent=[0,1,0,1])
cbar_ax.set_xticks([]); cbar_ax.set_yticks([])
for sp in cbar_ax.spines.values(): sp.set_edgecolor('#d1d5db'); sp.set_linewidth(0.5)
import matplotlib.patches as mpatches
dz_left = (MR - DZ) / (2 * MR)
dz_right = (MR + DZ) / (2 * MR)
cbar_ax.add_patch(mpatches.FancyBboxPatch((dz_left, 0), dz_right - dz_left, 1,
    boxstyle="square,pad=0", facecolor='none', edgecolor='#8a7a6a', linewidth=0, hatch='///', alpha=0.50))

fig.text(0.02, 0.075, '\u2190 Reverse tracking', fontsize=22, color='#dc2626', fontweight='600')
fig.text(0.50, 0.075, 'Culture-blind zone', fontsize=22, color='#92400e', fontweight='bold', ha='center')
fig.text(0.98, 0.075, 'Tracks correctly \u2192', fontsize=22, color='#2563eb', fontweight='600', ha='right')

fig.text(0.50, 0.028, "Values show Kendall\u2019s \u03C4. Cross-hatched cells fall within the culture-blind zone (|\u03C4| \u2264 0.10),",
         fontsize=18, color='#333333', ha='center')
fig.text(0.50, 0.003, "where a single pair flip can reverse the sign.",
         fontsize=18, color='#333333', ha='center')

plt.savefig('figure_45.png', dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.show()
print("Saved: figure_45.png")

#####  Merging Figure 3-5 into one file.


from PIL import Image
import io, re
from google.colab import files

def upload_img(prompt_name):
    print(f"Upload {prompt_name}:")
    uploaded = files.upload()
    base = prompt_name.replace('.png', '')
    for fname, content in uploaded.items():
        clean = re.sub(r'\s*\(\d+\)', '', fname.replace('.png', ''))
        if base.lower() in clean.lower():
            return Image.open(io.BytesIO(content))
    return Image.open(io.BytesIO(list(uploaded.values())[0]))

img3 = upload_img('figure_3.png')
img45 = upload_img('figure_45.png')

target_h = max(img3.height, img45.height)
img3r = img3.resize((int(img3.width * target_h / img3.height), target_h), Image.LANCZOS)
img45r = img45.resize((int(img45.width * target_h / img45.height), target_h), Image.LANCZOS)

border = 80
combined = Image.new('RGB', (img3r.width + border + img45r.width, target_h), (255, 255, 255))
combined.paste(img3r, (0, 0))
combined.paste(img45r, (img3r.width + border, 0))

combined.save('figure_345_combined.png', dpi=(300, 300))
display(combined)
print(f"Saved: figure_345_combined.png ({combined.width}x{combined.height})")
