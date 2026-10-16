"""Offline, seed-aware summaries of frozen-encoder evaluation results."""
from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
from pathlib import Path

from .settings import write_json


# Metric order also defines the initial sort within each feature readout.
METRICS = {
    'acquisition': {
        'transfer': ('Semantic transfer', True, True,
                     'Mean balanced accuracy across different train/test perturbation variants, using matched case folds. Higher is better.'),
        'cka': ('CKA', True, False,
                'Mean linear centered kernel alignment across different variant pairs. Higher means more similar feature geometry.'),
        'drift': ('Feature drift', False, False,
                  'Mean cosine distance from the reference variant. Lower means less representation change.'),
    },
    'cross_modal': {
        'map': ('Retrieval mAP', True, True, 'Mean average precision averaged over CT→MR and MR→CT retrieval.'),
        'top@1': ('Top-1', True, True, 'Fraction of queries with a same-organ result at rank 1, averaged over both retrieval directions.'),
        'top@5': ('Top-5', True, True, 'Fraction of queries with at least one same-organ result in the top 5, averaged over both retrieval directions.'),
        'modality_ilisi': ('Modality iLISI', True, False, 'Local modality mixing; higher indicates greater mixing within the supported cohort.'),
        'organ_clisi': ('Organ cLISI', False, False, 'Local anatomical mixing; lower indicates stronger organ separation.'),
    },
    'anatomical': {
        'probe_ba': ('Low-shot transfer', True, True, 'Logistic-readout balanced accuracy on case-disjoint query samples, macro-averaged across supported organ folds.'),
        'centroid_ba': ('Centroid baseline', True, True, 'Nearest-centroid balanced accuracy under the same evaluation protocol.'),
        'adaptation_gain': ('Adaptation gain', True, True, 'Balanced-accuracy gain over the nearest-centroid baseline, in percentage points.'),
    },
}
MODULE_NAMES = {'acquisition': 'Acquisition robustness', 'cross_modal': 'Cross-modal alignment',
                'anatomical': 'Anatomical generalization'}
FEATURE_NAMES = {'cls': 'CLS token', 'avg_pool': 'Average pooled patches', 'multilayer': 'Multilayer readout'}
COLUMNS = ['checkpoint', 'feature', 'module', 'metric', 'mean', 'std', 'n_seeds']
INTERPRETATION = ('Descriptive frozen-feature comparisons for this cohort and protocol. '
                  'No overall score or guaranteed downstream winner. '
                  'SD describes evaluation seeds, not pretraining runs.')


def _esc(value):
    return html.escape(str(value), quote=True)


def _checkpoint_key(name):
    from .core.checkpoints import resolve_checkpoint_config
    try:
        return resolve_checkpoint_config(name).checkpoint_key
    except KeyError:
        return name


def _summarize(cfg, rows):
    checkpoints = [_checkpoint_key(name) for name in cfg['checkpoints']]
    features, modules, seeds = cfg['feature_types'], cfg['modules'], cfg['seeds']
    for name, values in [('checkpoints', checkpoints), ('features', features), ('modules', modules), ('seeds', seeds)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Report {name} must be nonempty and unique')
    if not set(modules) <= METRICS.keys():
        raise ValueError('Report contains an unsupported module')
    groups, seen = {}, set()
    for row in rows:
        key = (row['checkpoint'], row['feature'], row['module'], row['metric'])
        identity = (*key, row['seed'])
        if identity in seen:
            raise ValueError(f'Duplicate report metric for seed: {identity}')
        if (key[0] not in checkpoints or key[1] not in features or key[2] not in modules
                or key[3] not in METRICS[key[2]] or row['seed'] not in seeds):
            raise ValueError(f'Unexpected report metric: {identity}')
        value = row['value']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Report metric must be finite: {identity}')
        seen.add(identity)
        groups.setdefault(key, {})[row['seed']] = value
    summary = []
    for key, values in sorted(groups.items()):
        summary.append(dict(zip(COLUMNS, [*key, statistics.mean(values.values()),
                       statistics.stdev(values.values()) if len(values) > 1 else None, len(values)])))
    coverage = []
    for module in modules:
        for feature in features:
            for checkpoint in checkpoints:
                missing = {metric: [seed for seed in seeds if seed not in groups.get((checkpoint, feature, module, metric), {})]
                           for metric in METRICS[module]}
                missing = {metric: values for metric, values in missing.items() if values}
                coverage.append({'checkpoint': checkpoint, 'feature': feature, 'module': module,
                                 'complete': not missing, 'missing_seeds_by_metric': missing})
    return summary, coverage, checkpoints


def _artifact_path(run_dir, row):
    if row.get('path'):
        path = Path(row['path'])
        return (path if path.is_absolute() else run_dir / path).resolve()
    suffix = Path('seeds') / f"seed_{row['seed']}" / f"{row['checkpoint']}--{row['feature']}" / f"{row['module']}.json"
    candidates = [run_dir / suffix, run_dir / row['module'] / suffix, run_dir / 'modules' / row['module'] / suffix]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _scope(module, data):
    if module == 'acquisition':
        semantic = data.get('semantic_metadata') or {}
        transfer = data.get('matched_semantic_transfer') or {}
        return {'source_volumes': data.get('n_source_volumes'), 'source_bin': data.get('source_bin'),
                'protocol': data.get('perturbation_protocol'), 'variants': data.get('target_variants'),
                'reference': data.get('reference_variant'),
                'labels': transfer.get('label_names', semantic.get('organ_names')), 'target_mode': semantic.get('mode'),
                'excluded_labels': [item.get('organ_name', str(item.get('organ_id'))) for item in semantic.get('excluded_labels', [])]}
    if module == 'cross_modal':
        return {'supported_organs': (data.get('cohorts') or {}).get('n_supported_organs')}
    transfer = (data.get('headline_metrics') or {}).get('task_2_low_shot_transfer') or {}
    return {key: transfer.get(key) for key in ('n_supported_organs', 'n_evaluated_organs', 'support_per_modality')}


def _evidence(run_dir, rows):
    evidence, seen = [], set()
    for row in rows:
        key = (row['checkpoint'], row['feature'], row['module'], row['seed'])
        if key in seen:
            continue
        seen.add(key)
        path = _artifact_path(run_dir, row)
        scope = _scope(row['module'], json.loads(path.read_text())) if path.is_file() else None
        evidence.append({'checkpoint': key[0], 'feature': key[1], 'module': key[2], 'seed': key[3],
                         'path': path, 'scope': scope})
    return evidence


def _scope_html(module, evidence):
    scopes = {json.dumps(item['scope'], sort_keys=True) for item in evidence if item['scope'] is not None}
    if not scopes:
        return '<p class="note">Cohort details are unavailable; consult the original evaluation artifacts.</p>'
    blocks = []
    for encoded in sorted(scopes):
        scope = json.loads(encoded)
        recorded = lambda value: _esc(value) if value is not None else 'not recorded'
        if module == 'acquisition':
            labels = ', '.join(scope['labels'] or []) or 'not recorded'
            variants = ' · '.join(scope['variants'] or []) or 'not recorded'
            source = scope['source_volumes'] if scope['source_volumes'] is not None else 'not recorded'
            text = (f'<strong>{_esc(source)} source volumes</strong> · variants: <strong>{_esc(variants)}</strong>'
                    f' · reference: {_esc(scope["reference"])}.'
                    f'<br>Semantic targets: <strong>{_esc(labels)}</strong>.')
            if scope['excluded_labels']:
                text += f' Excluded by label eligibility: {_esc(", ".join(scope["excluded_labels"]))}.'
            text += ' Semantic transfer describes these targets only.'
            if scope['target_mode'] in {'label_id_multilabel', 'segmentation_dir_presence', 'multi_binary'}:
                text += ' For binary presence targets, chance-level balanced accuracy is 50%.'
        elif module == 'cross_modal':
            text = f'Supported organs: <strong>{recorded(scope["supported_organs"])}</strong>. Retrieval is evaluated in both CT→MR and MR→CT directions.'
        else:
            text = (f'Evaluated organs: <strong>{recorded(scope["n_evaluated_organs"])}/{recorded(scope["n_supported_organs"])}</strong>'
                    f' · labeled support per modality: <strong>{recorded(scope["support_per_modality"])}</strong>.')
        blocks.append(f'<p>{text}</p>')
    if len(scopes) > 1:
        blocks.insert(0, '<p class="warning">Stored cohort or target scopes differ. Check the structured profile before comparing encoders.</p>')
    if any(item['scope'] is None for item in evidence):
        blocks.append('<p class="note">Some cohort details are unavailable in the stored artifacts.</p>')
    return '<div class="scope">' + ''.join(blocks) + '</div>'


def _metric_cell(row, specification):
    if row is None:
        return '<td class="missing">—</td>'
    factor = 100 if specification[2] else 1
    mean = row['mean'] * factor
    std = row['std']
    precision = 2 if factor == 100 else 4
    deviation = f'± {std * factor:.{precision}f}' if std is not None else 'SD unavailable'
    return (f'<td title="{_esc(row["mean"])}; SD: {_esc(std)}; n={row["n_seeds"]}">'
            f'<strong>{mean:.{precision}f}</strong> <span class="deviation">{deviation}</span></td>')


def _panel(module, feature, checkpoints, summary, coverage, evidence, allow_rank):
    specs = METRICS[module]
    primary = next(iter(specs))
    selected = {(row['checkpoint'], row['metric']): row for row in summary
                if row['module'] == module and row['feature'] == feature}
    panel_evidence = [row for row in evidence if row['module'] == module and row['feature'] == feature]
    complete = all(row['complete'] for row in coverage if row['module'] == module and row['feature'] == feature)
    scopes = {json.dumps(row['scope'], sort_keys=True) for row in panel_evidence if row['scope'] is not None}
    rankable = allow_rank and complete and len(scopes) <= 1
    ordered = sorted(checkpoints, key=lambda name: (-selected[name, primary]['mean'], name)) if rankable else checkpoints
    panel_id = f'{module}-{feature}'
    options = ''.join(f'<option value="{_esc(metric)}" data-direction="{"desc" if spec[1] else "asc"}">{_esc(spec[0])} {"↓" if spec[1] else "↑"}</option>'
                      for metric, spec in specs.items())
    controls = (f'<label class="sort-control" hidden>Sort by <select data-table="{_esc(panel_id)}">{options}</select></label>'
                if rankable else '<span class="note">Ranking withheld: incomplete results or differing scopes.</span>')
    headings = ''.join(
        f'<th scope="col" title="{_esc(spec[3])}">{_esc(spec[0])}'
        f'{" (pp)" if metric == "adaptation_gain" else " (%)" if spec[2] else ""} {"↑" if spec[1] else "↓"}</th>'
        for metric, spec in specs.items())
    body, previous, rank = [], None, 0
    for index, checkpoint in enumerate(ordered, 1):
        if rankable:
            value = selected[checkpoint, primary]['mean']
            if value != previous:
                rank = index
            previous = value
        attrs = ' '.join(f'data-{metric.replace("@", "-")}="{selected[checkpoint, metric]["mean"]}"'
                         for metric in specs if (checkpoint, metric) in selected)
        cells = ''.join(_metric_cell(selected.get((checkpoint, metric)), spec) for metric, spec in specs.items())
        body.append(f'<tr {attrs} data-checkpoint="{_esc(checkpoint)}"><td class="rank">{rank if rankable else "—"}</td>'
                    f'<th scope="row"><code>{_esc(checkpoint)}</code></th>{cells}</tr>')
    caption = (f'Mean ± sample SD across evaluation seeds · sorted by {_esc(specs[primary][0])}, highest first.' if rankable
               else 'Mean ± sample SD where available; — marks missing results. Rankings withheld.')
    hidden = ' hidden' if feature == 'multilayer' else ''
    return (f'<article class="panel"{hidden}><div class="panel-heading"><h3>{_esc(FEATURE_NAMES.get(feature, feature))}</h3>{controls}</div>'
            '<p class="scroll-hint note">Scroll horizontally to see all metrics.</p>'
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{_esc(FEATURE_NAMES.get(feature, feature))} comparison">'
            f'<table id="{_esc(panel_id)}"><caption>{caption}</caption><thead><tr><th scope="col">Rank</th><th scope="col">Encoder</th>{headings}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></article>')


STYLE = """
:root{color-scheme:light;--ink:#20332e;--muted:#596c64;--accent:#176b5b;--line:#dce4dd;--wash:#edf3ec}
*{box-sizing:border-box}body{margin:0;background:#f7f8f4;color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1080px;margin:auto;padding:32px 28px 48px}a{color:var(--accent);text-underline-offset:3px}
h1,h2,h3{line-height:1.25}h1{font-size:clamp(28px,4vw,42px);margin:12px 0}h2{font-size:26px;margin:0 0 12px}h3{font-size:19px;margin:0}
.eyebrow{color:var(--accent);font-weight:750;font-size:12px;letter-spacing:1.4px;text-transform:uppercase}
.intro{max-width:860px;color:var(--muted)}.badge{display:inline-block;padding:3px 11px;border-radius:20px;background:var(--wash);font-size:13px;margin-left:12px}
nav{display:flex;flex-wrap:wrap;gap:8px 22px;margin:16px 0;font-size:13px}.module{padding-top:26px;margin-top:24px;border-top:1px solid var(--line)}
.scope{padding:4px 16px;border-left:3px solid var(--accent);background:var(--wash);margin:16px 0;font-size:13px}.scope p{margin:8px 0}
.panel{background:white;border:1px solid var(--line);border-radius:8px;margin:18px 0;padding:16px}.panel-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.sort-control{font-size:12px;color:var(--muted)}select{font:inherit;color:var(--ink);padding:5px 7px;background:white;border:1px solid #acbfb3;border-radius:5px;margin-left:5px}
.table-scroll{overflow-x:auto;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);font-size:12px;margin:0 0 10px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:middle}thead th{font-size:12px;white-space:nowrap;background:var(--wash)}tbody th{font-weight:600;min-width:180px}tbody tr:last-child>*{border-bottom:0}td{white-space:nowrap}td strong{font-weight:650}.deviation{color:var(--muted);font-size:12px}code{font-size:12px;overflow-wrap:anywhere}
.rank{color:var(--accent)}.note,.missing{color:var(--muted);font-size:13px}.warning{padding:12px 16px;border-left:3px solid #a66b24;background:#fbf1df;color:#634015}
details{margin-top:18px}summary{cursor:pointer;color:var(--accent);font-weight:650}.metric-guide{margin-top:22px}.metric-guide h3{font-size:16px}.definitions{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px 24px;margin:12px 0}.definitions dt{font-size:13px;font-weight:650}.definitions dd{margin:4px 0 0;color:var(--muted);font-size:13px}
footer{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}[hidden]{display:none!important}
.scroll-hint{display:none}:focus-visible{outline:3px solid var(--accent);outline-offset:4px}@media(max-width:650px){main{padding:24px 16px}.panel{padding:14px 10px}th,td{padding:8px}.scope{padding:4px 12px}.scroll-hint{display:block;margin:10px 0 0}}
@media print{body{background:white}main{max-width:none;padding:0}.sort-control{display:none}.table-scroll{overflow:visible}.panel{break-inside:avoid}a{color:inherit}}
"""

SCRIPT = """
document.querySelectorAll('.sort-control').forEach(control => {
  control.hidden = false;
  const select = control.querySelector('select');
  select.addEventListener('change', () => {
    const table = document.getElementById(select.dataset.table);
    const attribute = 'data-' + select.value.replace('@', '-');
    const direction = select.selectedOptions[0].dataset.direction === 'asc' ? 1 : -1;
    const rows = [...table.tBodies[0].rows];
    rows.sort((a, b) => direction * (Number(a.getAttribute(attribute)) - Number(b.getAttribute(attribute)))
      || a.dataset.checkpoint.localeCompare(b.dataset.checkpoint));
    let previous = null, rank = 0;
    rows.forEach((row, index) => {
      const value = Number(row.getAttribute(attribute));
      if (value !== previous) rank = index + 1;
      row.querySelector('.rank').textContent = rank;
      previous = value;
      table.tBodies[0].appendChild(row);
    });
    table.caption.textContent = 'Mean ± sample SD across evaluation seeds · sorted by '
      + select.selectedOptions[0].textContent + '.';
  });
});
"""


def write_report(run_dir, cfg, rows, status='complete'):
    """Write scores.csv, profile.json and a self-contained report without inference.

    Rows contain checkpoint, feature, module, metric, seed and value. An optional
    path points to the corresponding module JSON when it is outside the standard
    seed directory. Missing evaluations remain visible and suppress rankings.
    """
    run_dir = Path(run_dir)
    rows = list(rows)
    summary, coverage, checkpoints = _summarize(cfg, rows)
    evidence = _evidence(run_dir, rows)
    complete = status == 'complete' and all(row['complete'] for row in coverage)
    status = 'complete' if complete else 'incomplete'
    best, by_feature = {}, {}
    if complete:
        for row in summary:
            scopes = {json.dumps(item['scope'], sort_keys=True) for item in evidence
                      if item['module'] == row['module'] and item['scope'] is not None}
            if len(scopes) > 1:
                continue
            for collection, key in [(best, f'{row["module"]}/{row["metric"]}'),
                                    (by_feature, f'{row["module"]}/{row["feature"]}/{row["metric"]}')]:
                current = collection.get(key)
                higher = METRICS[row['module']][row['metric']][1]
                if current is None or (row['mean'] > current['mean'] if higher else row['mean'] < current['mean']):
                    collection[key] = row
    profile = {'status': status, 'dataset': cfg['dataset'], 'not_evaluated': cfg.get('not_evaluated', []),
               'metrics': summary, 'highest_observed_per_property': best, 'interpretation': INTERPRETATION,
               'highest_observed_per_feature': by_feature, 'coverage': coverage,
               'expected_seeds': cfg['seeds'],
               'evidence': [{**item, 'path': os.path.relpath(item['path'], run_dir)} for item in evidence]}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(summary)
    write_json(run_dir / 'profile.json', profile)
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>REFORM-3D · {_esc(cfg["dataset"])}</title><style>{STYLE}</style></head><body><main>'
             f'<header><div class="eyebrow">REFORM-3D · Frozen-encoder assessment</div><h1>{_esc(cfg["dataset"])}'
             f'<span class="badge">{_esc(status)}</span></h1><p class="intro">Compare encoder readouts before downstream adaptation. '
             'Each table keeps one feature type and protocol together; choose the property relevant to your application.</p>'
             '<nav aria-label="Report downloads"><a href="scores.csv">Download scores CSV</a>'
             '<a href="profile.json">Structured profile</a>'
             + ('<a href="run.json">Run provenance</a>' if (run_dir / 'run.json').is_file() else '') + '</nav></header>']
    if not complete:
        parts.append('<p class="warning">Incomplete evaluation. Available values are shown for inspection; rankings are withheld until every configured encoder, feature, metric and seed is complete.</p>')
    for module in cfg['modules']:
        module_evidence = [row for row in evidence if row['module'] == module]
        parts.append(f'<section class="module" id="{_esc(module)}"><h2>{MODULE_NAMES[module]}</h2>')
        parts.append(_scope_html(module, module_evidence))
        primary = [feature for feature in ('cls', 'avg_pool') if feature in cfg['feature_types']]
        secondary = [feature for feature in cfg['feature_types'] if feature not in primary]
        for feature in primary + secondary:
            parts.append(_panel(module, feature, checkpoints, summary, coverage, evidence, complete))
        definitions = ''.join(f'<div><dt>{_esc(spec[0])} {"↑" if spec[1] else "↓"}</dt><dd>{_esc(spec[3])}</dd></div>' for spec in METRICS[module].values())
        parts.append(f'<div class="metric-guide"><h3>How to read these metrics</h3><dl class="definitions">{definitions}</dl>')
        if module == 'acquisition':
            parts.append('<p class="note">Compare semantic transfer alongside CKA and drift: stable features alone do not imply useful information. '
                         'These results describe the tested resampling variants and semantic targets.</p>')
        parts.append('</div></section>')
    if not complete:
        missing = [row for row in coverage if not row['complete']]
        parts.append('<details><summary>Missing evaluations</summary><ul>' + ''.join(
            f'<li>{_esc(row["checkpoint"])} / {_esc(row["feature"])} / {_esc(row["module"])}: {_esc(json.dumps(row["missing_seeds_by_metric"]))}</li>'
            for row in missing) + '</ul></details>')
    parts.append(f'<footer><p>{_esc(INTERPRETATION)}</p><p>Not evaluated: {_esc(", ".join(profile["not_evaluated"]) or "none")}.</p>'
                 '</footer></main><script>' + SCRIPT + '</script></body></html>')
    (run_dir / 'report.html').write_text(''.join(parts), encoding='utf-8')
    return profile
