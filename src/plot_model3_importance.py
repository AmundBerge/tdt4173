"""Render a compact report figure from interpret_model3.py's saved results."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from interpret_model3 import readable, plot_detailed, feature_group


def main():
    output = Path('results/model3_interpretation')
    mean = pd.read_csv(output / 'mean_feature_importance.csv', index_col=0).iloc[:, 0]
    top = mean.head(10).iloc[::-1] * 100
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(left=0.43, right=0.93, top=0.80, bottom=0.18)
    bars = ax.barh([readable(c) for c in top.index], top, color='#227c9d', height=0.65)
    ax.bar_label(bars, fmt='%.2f%%', padding=5, fontsize=10)
    ax.set_xlim(0, top.max() * 1.25)
    ax.set_xlabel('Mean normalized impurity importance (%)', labelpad=10)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=8)
    fig.suptitle('Feature importance in the final random forest models',
                 x=0.05, y=0.96, ha='left', weight='bold', fontsize=14)
    fig.text(0.05, 0.89, 'Top 10 features · equal-weight average across 14 generating units', fontsize=10)
    fig.text(0.05, 0.035,
             'Final model configuration fitted on all labeled 2015–2022 cases.\n'
             'Unused features count as zero; Lio averages both forests. Importance does not imply causality.',
             fontsize=8.5, color='#46515b')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'model3_report_importance.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    importance = pd.read_csv(output / 'feature_importance_by_unit.csv', index_col=0)
    grouped = importance.groupby([feature_group(c) for c in importance.index], sort=False).sum()
    grouped = grouped.loc[grouped.mean(axis=1).sort_values(ascending=False).index]
    grouped.to_csv(output / 'group_importance_by_unit.csv')
    plot_detailed(output, mean, grouped, list(grouped.columns))


if __name__ == '__main__':
    main()
