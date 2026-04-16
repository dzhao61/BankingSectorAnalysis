"""
Generates all README figures from pre-computed CSV results.
Saves PNGs to the images/ directory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs('images', exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
mi_ts       = pd.read_csv('mi_time_series.csv',    index_col=0, parse_dates=True)
cmi_both    = pd.read_csv('cmi_both_controls.csv', index_col=0, parse_dates=True)
cmi_interest= pd.read_csv('cmi_interest_rates.csv',index_col=0, parse_dates=True)
cmi_asx200  = pd.read_csv('cmi_asx200.csv',        index_col=0, parse_dates=True)
mi_pvals    = pd.read_csv('mi_pvalues.csv',        index_col=0, parse_dates=True)
cmi_pvals   = pd.read_csv('cmi_pvalues.csv',       index_col=0, parse_dates=True)

PLOT_START = pd.to_datetime('2015-01-01')
X_FMT = '%b\n%Y'

events = {
    '2015-06-01': 'China Slowdown',
    '2017-02-01': 'Trump 1st Term',
    '2018-06-01': 'Banking Royal Commission',
    '2020-03-01': 'COVID-19',
    '2022-05-15': 'Inflation + RBA Hikes',
    '2023-03-01': 'SVB Crisis',
    '2024-11-01': 'Trump 2nd Term',
}

pair_cols    = [c for c in mi_ts.columns]
pairs_sorted = mi_ts[pair_cols].mean(axis=0).sort_values(ascending=False).index.tolist()

def add_events(ax, index):
    ymax = ax.get_ylim()[1]
    for ds, label in events.items():
        d = pd.to_datetime(ds)
        if index[0] <= d <= index[-1]:
            ax.axvline(d, color='black', ls='--', lw=1, alpha=0.4)
            ax.text(d, ymax * 0.99, ' ' + label, rotation=90, fontsize=7.5,
                    color='black', ha='left', va='top', fontweight='bold')

def set_date_ticks(ax, index):
    ax.set_xlim(index[0], index[-1])
    ticks = pd.date_range(start='2015-01-01', end=index[-1], freq='6MS')
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(X_FMT))


# ── Figure 1: Mean MI across conditioning factors ─────────────────────────
def filt(s): return s[s.index >= PLOT_START]

mean_mi    = filt(mi_ts[pair_cols].mean(axis=1))
mean_cmi_i = filt(cmi_interest.mean(axis=1))
mean_cmi_m = filt(cmi_asx200.mean(axis=1))
mean_cmi_b = filt(cmi_both.mean(axis=1))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(mean_mi.index,    mean_mi.values,    color='black',   lw=2, label='MI (unconditional)')
ax.plot(mean_cmi_i.index, mean_cmi_i.values, color='#1f77b4', lw=2, label='CMI | Aus. Govt. Bond Index')
ax.plot(mean_cmi_m.index, mean_cmi_m.values, color='#2ca02c', lw=2, label='CMI | ASX200 Ex-Top 20 Index')
ax.plot(mean_cmi_b.index, mean_cmi_b.values, color='#d62728', lw=2, label='CMI | Both Variables')
ax.set_ylabel('Mean Mutual Information (nats)', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_title('Mean Mutual Information Across Conditioning Factors (2015–2025)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
set_date_ticks(ax, mean_mi.index)
add_events(ax, mean_mi.index)
plt.tight_layout()
plt.savefig('images/mi_conditioning_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: images/mi_conditioning_comparison.png')


# ── Figure 2: MI heatmap (value + significance) ───────────────────────────
mi_f   = mi_ts[mi_ts.index >= PLOT_START][pairs_sorted]
cmi_f  = cmi_both[cmi_both.index >= PLOT_START][pairs_sorted]
mip_f  = mi_pvals[mi_pvals.index >= PLOT_START][pairs_sorted]
shared_vmax = max(mi_f.values.max(), cmi_f.values.max())

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for row, (data, title, cbar_label, cmap, vmin, vmax) in enumerate([
    (mi_f,                  'Mutual Information Values (nats)',              'MI (nats)',   'viridis',  0, shared_vmax),
    ((mip_f < 0.05).astype(int), 'Statistical Significance (Green = p < 0.05)', 'Significant', 'RdYlGn', 0, 1),
]):
    ax = axes[row]
    plot_dates = data.index
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(pairs_sorted)))
    ax.set_yticklabels(pairs_sorted, fontsize=9, fontweight='bold')
    ax.set_ylabel('Bank Pair', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    dr = pd.date_range(start='2015-01-01', end=plot_dates[-1], freq='6MS')
    pos  = [np.argmin(np.abs(plot_dates - d)) for d in dr]
    labs = [d.strftime(X_FMT) for d in dr]
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9, fontweight='bold')
    if row == 1:
        cbar.set_ticks([0, 1]); ax.set_xlabel('Date', fontsize=10, fontweight='bold')
plt.suptitle('Pairwise Mutual Information — Sorted by Average MI', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/mi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: images/mi_heatmap.png')


# ── Figure 3: CMI heatmap (value + significance) ─────────────────────────
cmip_f = cmi_pvals[cmi_pvals.index >= PLOT_START][pairs_sorted]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for row, (data, title, cbar_label, cmap, vmin, vmax) in enumerate([
    (cmi_f,                   'Conditional Mutual Information Values (nats)',    'CMI (nats)',   'viridis', 0, shared_vmax),
    ((cmip_f < 0.05).astype(int), 'Statistical Significance (Green = p < 0.05)', 'Significant', 'RdYlGn', 0, 1),
]):
    ax = axes[row]
    plot_dates = data.index
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(pairs_sorted)))
    ax.set_yticklabels(pairs_sorted, fontsize=9, fontweight='bold')
    ax.set_ylabel('Bank Pair', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    dr = pd.date_range(start='2015-01-01', end=plot_dates[-1], freq='6MS')
    pos  = [np.argmin(np.abs(plot_dates - d)) for d in dr]
    labs = [d.strftime(X_FMT) for d in dr]
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9, fontweight='bold')
    if row == 1:
        cbar.set_ticks([0, 1]); ax.set_xlabel('Date', fontsize=10, fontweight='bold')
plt.suptitle('Conditional MI (conditioned on Market + Bond Index) — Same Scale as MI', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/cmi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: images/cmi_heatmap.png')


# ── Figure 4: Mean MI per bank ─────────────────────────────────────────────
bank_names = ['CBA', 'WBC', 'ANZ', 'NAB', 'MQG']
colors     = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bank_mi = pd.DataFrame(index=mi_ts.index)
for bank in bank_names:
    cols = [c for c in mi_ts.columns if bank in c.split('-')]
    bank_mi[bank] = mi_ts[cols].mean(axis=1)
bank_mi_f = bank_mi[bank_mi.index >= PLOT_START]

fig, ax = plt.subplots(figsize=(14, 5))
for i, bank in enumerate(bank_names):
    ax.plot(bank_mi_f.index, bank_mi_f[bank].values, color=colors[i], lw=2, label=bank)
ax.set_ylabel('Mean MI with Other Banks (nats)', fontsize=11, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_title('Mean MI per Bank — Average Across All Pairs Involving That Bank', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
set_date_ticks(ax, bank_mi_f.index)
add_events(ax, bank_mi_f.index)
plt.tight_layout()
plt.savefig('images/mi_per_bank.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: images/mi_per_bank.png')


# ── Figure 5: MI vs CMI summary bar charts ────────────────────────────────
summary = []
for pair in pairs_sorted:
    summary.append({
        'Pair':     pair,
        'MI Mean':  mi_ts[pair].mean(),
        'CMI Mean': cmi_both[pair].mean(),
        'MI Sig%':  100 * (mi_pvals[pair] < 0.05).sum() / len(mi_pvals),
        'CMI Sig%': 100 * (cmi_pvals[pair] < 0.05).sum() / len(cmi_pvals),
    })
sdf = pd.DataFrame(summary).sort_values('MI Mean', ascending=True)
x, w = np.arange(len(sdf)), 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(x - w/2, sdf['MI Mean'],  w, label='MI (unconditional)', color='#1f77b4')
ax.barh(x + w/2, sdf['CMI Mean'], w, label='CMI (conditioned)',  color='#d62728')
ax.set_yticks(x); ax.set_yticklabels(sdf['Pair'], fontsize=9)
ax.set_xlabel('Mean Value (nats)', fontsize=10, fontweight='bold')
ax.set_title('Average MI vs CMI per Bank Pair', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='x')

ax = axes[1]
ax.barh(x - w/2, sdf['MI Sig%'],  w, label='MI significant',  color='#1f77b4')
ax.barh(x + w/2, sdf['CMI Sig%'], w, label='CMI significant', color='#d62728')
ax.set_yticks(x); ax.set_yticklabels(sdf['Pair'], fontsize=9)
ax.set_xlabel('% of Time Significant (p < 0.05)', fontsize=10, fontweight='bold')
ax.set_title('% Time Statistically Significant per Bank Pair', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('MI vs CMI Summary by Bank Pair', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/significance_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: images/significance_summary.png')

print('\nAll figures saved to images/')
