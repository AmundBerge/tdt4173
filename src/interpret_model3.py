"""Reproduce final model3 forests and plot their impurity-based importances.

Run from the project root: .venv/Scripts/python.exe src/interpret_model3.py
Uses the existing notebook feature builder and all labeled training cases.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def feature_group(name):
    for prefixes, group in [
        (('margin_water_end_',), 'Hourly price–water margin'),
        (('weekly_margin_', 'profitable_share_', 'local_profitable_'), 'Weekly / nearby profitability'),
        (('volume_',), 'Initial storage'),
        (('weekly_inflow_', 'inflow_ratio_', 'headroom_'), 'Extra water availability'),
        (('inflow_',), 'Hourly inflow'),
        (('min_volume_', 'min_flow_'), 'Environmental constraints'),
    ]:
        if name.startswith(prefixes):
            return group
    if name in ('hour', 'day', 'month', 'day_of_year', 'horizon_hour'):
        return 'Calendar / horizon position'
    return 'Relative price / price shape'


def readable(name):
    for prefix, label in [
        ('margin_water_end_', 'Hourly margin'), ('weekly_margin_', 'Weekly mean margin'),
        ('profitable_share_', 'Profitable-hour share'),
        ('local_profitable_', 'Nearby profitable-hour share'),
        ('volume_', 'Initial storage'), ('inflow_', 'Hourly inflow'),
    ]:
        if name.startswith(prefix):
            return f'{label}: {name[len(prefix):]}'
    return name.replace('_', ' ').capitalize()


def plot_detailed(output, mean, grouped, units):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(16, 6.6),
                                   gridspec_kw={'width_ratios': [1, 1.35]})
    fig.subplots_adjust(left=0.22, right=0.96, top=0.84, bottom=0.24, wspace=0.88)
    top = mean.head(12).iloc[::-1]
    bars = ax.barh([readable(c) for c in top.index], top.values * 100, color='#227c9d')
    ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
    ax.set_xlim(0, top.max() * 118)
    ax.set_xlabel('Mean normalized importance (%)')
    ax.set_title('(a) Leading individual features', loc='left', pad=16, weight='bold')
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    values = grouped.to_numpy() * 100
    im = heat.imshow(values, cmap='Blues', aspect='auto', vmin=0, vmax=max(50, values.max()))
    heat.set_yticks(range(len(grouped)), grouped.index)
    heat.set_xticks(range(len(units)), [u.replace('_G', ' G') for u in units], rotation=65, ha='right')
    heat.tick_params(axis='both', labelsize=9)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            heat.text(c, r, f'{values[r,c]:.0f}', ha='center', va='center', fontsize=8,
                      color='white' if values[r,c] > im.norm.vmax * 0.55 else '#142c40')
    heat.set_title('(b) Feature groups by generating unit', loc='left', pad=16, weight='bold')
    fig.colorbar(im, ax=heat, fraction=0.035, pad=0.025, label='Normalized importance (%)')
    fig.suptitle('Random forest interpretation: final model configuration',
                 fontsize=17, weight='bold', y=0.96)
    fig.text(0.5, 0.905, 'Impurity-based importance · fitted on all labeled 2015–2022 cases',
             ha='center', fontsize=11, color='#46515b')
    fig.text(0.03, 0.035,
             'Panel (a): equal-weight mean across 14 units; unused features contribute zero. '
             'Panel (b): sum within each feature group.\n'
             'Lio combines both forests with equal weights. Importances describe tree splits; '
             'they do not show effect direction or causality.', fontsize=9, color='#46515b')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'model3_feature_importance.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def main():
    output = Path('results/model3_interpretation')
    output.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(Path('model3short.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('reservoirs ='):
            exec(source.split("test = pd.read_csv")[0], namespace)
            break
        exec(source, namespace)
    train = namespace['train']
    X = namespace['make_features'](train)
    targets = namespace['target_columns']
    y = train[targets].to_numpy(dtype=np.uint8).reshape(-1, 14, 168)
    y = y.transpose(0, 2, 1).reshape(-1, 14)
    units = [c.removeprefix('result_committed_').rsplit('_t', 1)[0] for c in targets[::168]]
    core = [c for c in X if not c.startswith(('weekly_inflow_', 'inflow_ratio_', 'headroom_'))]
    shared = [c for c in core if not c.startswith(
        ('volume_', 'inflow_', 'margin_', 'weekly_margin_', 'profitable_share_', 'local_profitable_'))]
    importance = pd.DataFrame(0.0, index=X.columns, columns=units)
    print(f'{len(train)} cases, {len(X)} hourly rows, {len(X.columns)} features', flush=True)
    def fit_unit(j):
        plant = namespace['plants'][j]
        local = [c for c in core if c in shared or any(
            c.endswith('_' + name) for name in namespace['reservoirs'][plant])]
        columns = core if plant in ('Hogga', 'Lio', 'Kjela') else local
        settings = dict(n_estimators=40, max_depth=20, min_samples_leaf=5,
                        max_samples=0.25, max_features=0.7)
        if plant == 'Haukeli':
            columns = list(X.columns)
            settings = dict(n_estimators=60, min_samples_leaf=2, max_samples=0.35)
        elif plant == 'Vinje':
            settings = dict(n_estimators=100, max_samples=0.5)
        model = RandomForestClassifier(random_state=67, n_jobs=1, **settings)
        model.fit(X[columns], y[:, j])
        result = pd.Series(0.0, index=X.columns, name=units[j])
        result.loc[columns] = model.feature_importances_
        if plant == 'Lio':
            deeper = RandomForestClassifier(n_estimators=100, max_samples=0.5,
                                             random_state=67, n_jobs=1)
            deeper.fit(X[local], y[:, j])
            result *= 0.5
            result.loc[local] += 0.5 * deeper.feature_importances_
        result.to_csv(output / f'importance_{units[j]}.csv')
        print(f'Finished {units[j]} ({len(columns)} primary features)', flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        importance = pd.concat(list(pool.map(fit_unit, range(14))), axis=1)
    assert np.allclose(importance.sum(axis=0), 1)
    importance.to_csv(output / 'feature_importance_by_unit.csv')
    mean = importance.mean(axis=1).sort_values(ascending=False)
    mean.rename('mean_importance').to_csv(output / 'mean_feature_importance.csv')
    grouped = importance.groupby([feature_group(c) for c in importance.index], sort=False).sum()
    grouped = grouped.loc[grouped.mean(axis=1).sort_values(ascending=False).index]
    grouped.to_csv(output / 'group_importance_by_unit.csv')

    plot_detailed(output, mean, grouped, units)
    metadata = dict(cases=len(train), hourly_rows=len(X), core_features=len(core),
                    generated_features=len(X.columns), units=units,
                    training_start=str(train.starttime.min()), training_end=str(train.starttime.max()),
                    method='Normalized mean decrease in impurity; equal unit weighting; Lio 50/50 blend')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Top features:\n' + mean.head(12).to_string(), flush=True)
    print('Mean group importance:\n' + grouped.mean(axis=1).to_string(), flush=True)


if __name__ == '__main__':
    main()

