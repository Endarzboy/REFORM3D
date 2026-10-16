import csv
import json
import math
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from reform3d.reporting import METRICS, SCRIPT, write_report


def configuration(**changes):
    cfg = {'dataset': 'example_ct', 'checkpoints': ['encoder_a', 'encoder_b'],
           'feature_types': ['cls', 'avg_pool', 'multilayer'], 'modules': ['acquisition'],
           'seeds': [42, 123], 'not_evaluated': ['cross_modal', 'anatomical']}
    return {**cfg, **changes}


def measurements(cfg):
    return [{'checkpoint': checkpoint, 'feature': feature, 'module': module, 'metric': metric,
             'seed': seed, 'value': 0.6 + checkpoint_index * 0.1 + seed_index * 0.02}
            for checkpoint_index, checkpoint in enumerate(cfg['checkpoints'])
            for feature in cfg['feature_types'] for module in cfg['modules']
            for metric in METRICS[module] for seed_index, seed in enumerate(cfg['seeds'])]


class Tables(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tables = {}
        self.hidden_tables = set()
        self.elements = []
        self.current = None
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag not in {'meta', 'br', 'link', 'input', 'hr', 'img'}:
            self.elements.append((tag, 'hidden' in attrs))
        if tag == 'table':
            self.current = attrs['id']
            self.tables[self.current] = []
            if any(hidden for _, hidden in self.elements):
                self.hidden_tables.add(self.current)
        if tag == 'tr' and 'data-checkpoint' in attrs:
            self.tables[self.current].append(attrs['data-checkpoint'])
        if tag == 'a':
            self.links.append(attrs.get('href'))

    def handle_endtag(self, tag):
        while self.elements:
            if self.elements.pop()[0] == tag:
                break


def add_evidence(tmp_path, rows, **changes):
    payload = {'n_source_volumes': 69, 'source_bin': 0, 'perturbation_protocol': 'fixed_absolute',
               'target_variants': ['1x1x1', '1x1x3', '1x1x5'], 'reference_variant': '1x1x1',
               'matched_semantic_transfer': {'label_names': ['cyst']},
               'semantic_metadata': {'excluded_labels': [{'organ_name': 'kidney'}, {'organ_name': 'tumor'}]}}
    payload.update(changes)
    for row in rows:
        leaf = tmp_path / 'seeds' / f'seed_{row["seed"]}' / f'{row["checkpoint"]}--{row["feature"]}'
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / f'{row["module"]}.json').write_text(json.dumps(payload))


def test_report_retains_precision_and_sample_seed_deviation(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    rows[0]['value'] = 0.6123456789123456
    profile = write_report(tmp_path, cfg, rows)
    indexed = {(row['checkpoint'], row['feature'], row['module'], row['metric']): row for row in profile['metrics']}
    observed = indexed['encoder_a', 'cls', 'acquisition', 'transfer']
    assert observed['mean'] == (0.6123456789123456 + 0.62) / 2
    assert observed['std'] == pytest.approx(abs(0.6123456789123456 - 0.62) / math.sqrt(2))
    assert observed['n_seeds'] == 2
    with (tmp_path / 'scores.csv').open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['checkpoint', 'feature', 'module', 'metric', 'mean', 'std', 'n_seeds']
        saved = list(reader)
    assert any(float(row['mean']) == observed['mean'] for row in saved)
    assert json.loads((tmp_path / 'profile.json').read_text()) == profile


def test_rankings_keep_readouts_separate_and_lower_drift_wins(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    for row in rows:
        if row['checkpoint'] == 'encoder_a' and row['feature'] == 'avg_pool' and row['metric'] == 'transfer':
            row['value'] = 0.9
    profile = write_report(tmp_path, cfg, rows)
    tables = Tables((tmp_path / 'report.html').read_text()).tables
    assert tables['acquisition-cls'] == ['encoder_b', 'encoder_a']
    assert tables['acquisition-avg_pool'] == ['encoder_a', 'encoder_b']
    assert profile['highest_observed_per_feature']['acquisition/cls/transfer']['checkpoint'] == 'encoder_b'
    assert profile['highest_observed_per_feature']['acquisition/avg_pool/transfer']['checkpoint'] == 'encoder_a'
    assert profile['highest_observed_per_property']['acquisition/drift']['checkpoint'] == 'encoder_a'
    html = (tmp_path / 'report.html').read_text()
    assert 'data-direction="asc">Feature drift ↑' in html
    parsed = Tables(html)
    assert parsed.hidden_tables == {'acquisition-multilayer'}
    assert parsed.tables['acquisition-multilayer'] == ['encoder_b', 'encoder_a']
    assert 'Additional readout' not in html
    assert 'acquisition/multilayer/transfer' in profile['highest_observed_per_feature']


def test_compact_report_keeps_units_and_inline_uncertainty(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    add_evidence(tmp_path, rows)
    write_report(tmp_path, cfg, rows)
    html = (tmp_path / 'report.html').read_text()
    assert '<strong>61.00</strong> <span class="deviation">± 1.41</span>' in html
    assert 'Semantic transfer (%) ↑' in html
    assert '<h3>How to read these metrics</h3>' in html
    assert '<summary>How to read these metrics' not in html
    assert 'metric-bar' not in html
    assert 'Complete encoder / feature / module groups' not in html
    assert '<dt>Encoders</dt>' not in html and '<dt>Evaluation seeds</dt>' not in html
    assert 'class="stats"' not in html


def test_missing_metric_seed_cannot_claim_complete_profile_or_rank(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    rows = [row for row in rows if not (row['checkpoint'] == 'encoder_a' and row['feature'] == 'cls'
                                      and row['metric'] == 'transfer' and row['seed'] == 123)]
    profile = write_report(tmp_path, cfg, rows)
    assert profile['status'] == 'incomplete'
    assert profile['highest_observed_per_property'] == {}
    assert profile['highest_observed_per_feature'] == {}
    missing = [row for row in profile['coverage'] if not row['complete']]
    assert missing == [{'checkpoint': 'encoder_a', 'feature': 'cls', 'module': 'acquisition',
                        'complete': False, 'missing_seeds_by_metric': {'transfer': [123]}}]
    html = (tmp_path / 'report.html').read_text()
    assert '<td class="rank">—</td>' in html
    assert '<label class="sort-control"' not in html
    assert 'Incomplete evaluation' in html


def test_missing_entire_encoder_is_visible(tmp_path):
    cfg = configuration()
    rows = [row for row in measurements(cfg) if row['checkpoint'] == 'encoder_a']
    profile = write_report(tmp_path, cfg, rows)
    assert profile['status'] == 'incomplete'
    tables = Tables((tmp_path / 'report.html').read_text()).tables
    assert tables['acquisition-cls'] == ['encoder_a', 'encoder_b']
    assert '<td class="missing">—</td>' in (tmp_path / 'report.html').read_text()


def test_hidden_multilayer_still_counts_toward_completeness(tmp_path):
    cfg = configuration()
    rows = [row for row in measurements(cfg) if row['feature'] != 'multilayer']
    profile = write_report(tmp_path, cfg, rows)
    html = (tmp_path / 'report.html').read_text()
    assert profile['status'] == 'incomplete'
    assert profile['highest_observed_per_feature'] == {}
    assert {row['feature'] for row in profile['coverage'] if not row['complete']} == {'multilayer'}
    assert Tables(html).hidden_tables == {'acquisition-multilayer'}
    assert '<label class="sort-control"' not in html


def test_duplicate_metric_rejected_before_overwriting_report(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    target = tmp_path / 'report.html'
    target.write_text('keep existing report')
    with pytest.raises(ValueError, match='Duplicate report metric'):
        write_report(tmp_path, cfg, [*rows, rows[0]])
    assert target.read_text() == 'keep existing report'
    assert not (tmp_path / 'scores.csv').exists()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, None])
def test_nonfinite_or_invalid_metric_rejected(tmp_path, value):
    cfg = configuration()
    rows = measurements(cfg)
    rows[0]['value'] = value
    with pytest.raises(ValueError, match='must be finite'):
        write_report(tmp_path, cfg, rows)


def test_unexpected_seed_cannot_substitute_for_configured_seed(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    rows[0]['seed'] = 456
    with pytest.raises(ValueError, match='Unexpected report metric'):
        write_report(tmp_path, cfg, rows)


def test_one_seed_has_no_fabricated_sd(tmp_path):
    cfg = configuration(seeds=[42])
    profile = write_report(tmp_path, cfg, measurements(cfg))
    assert all(row['std'] is None and row['n_seeds'] == 1 for row in profile['metrics'])
    assert 'SD unavailable' in (tmp_path / 'report.html').read_text()


def test_empty_and_failed_runs_never_have_rankings(tmp_path):
    cfg = configuration()
    empty = write_report(tmp_path, cfg, [])
    assert empty['status'] == 'incomplete'
    assert empty['metrics'] == []
    failed = write_report(tmp_path, cfg, measurements(cfg), status='incomplete')
    assert failed['highest_observed_per_property'] == {}
    assert 'rank">—' in (tmp_path / 'report.html').read_text()


def test_scope_uses_actual_labels_counts_and_variants(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    add_evidence(tmp_path, rows)
    profile = write_report(tmp_path, cfg, rows)
    html = (tmp_path / 'report.html').read_text()
    assert '69 source volumes' in html
    assert 'Semantic targets: <strong>cyst</strong>' in html
    assert 'Excluded by label eligibility: kidney, tumor.' in html
    assert '1x1x1 · 1x1x3 · 1x1x5' in html
    assert profile['evidence'][0]['scope']['labels'] == ['cyst']
    # Scope comes from artifacts, rather than the dataset name or hardcoded KiTS values.
    add_evidence(tmp_path, rows, n_source_volumes=23, matched_semantic_transfer={'label_names': ['aorta']},
                 target_variants=['native', 'z2'], semantic_metadata={})
    write_report(tmp_path, cfg, rows)
    html = (tmp_path / 'report.html').read_text()
    assert '23 source volumes' in html and '<strong>aorta</strong>' in html
    assert 'cyst' not in html and 'kidney' not in html


def test_differing_scope_withholds_encoder_rankings(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    add_evidence(tmp_path, rows)
    add_evidence(tmp_path, [row for row in rows if row['checkpoint'] == 'encoder_b'], n_source_volumes=2)
    profile = write_report(tmp_path, cfg, rows)
    assert profile['highest_observed_per_property'] == {}
    html = (tmp_path / 'report.html').read_text()
    assert 'Stored cohort or target scopes differ' in html
    assert '<label class="sort-control"' not in html


def test_all_three_modules_have_directional_metric_tables(tmp_path):
    cfg = configuration(modules=['acquisition', 'cross_modal', 'anatomical'], not_evaluated=[])
    profile = write_report(tmp_path, cfg, measurements(cfg))
    html = (tmp_path / 'report.html').read_text()
    parsed = Tables(html)
    assert len(parsed.tables) == 9
    assert parsed.hidden_tables == {f'{module}-multilayer' for module in cfg['modules']}
    assert 'Cross-modal alignment' in html and 'Anatomical generalization' in html
    assert profile['highest_observed_per_property']['cross_modal/organ_clisi']['checkpoint'] == 'encoder_a'
    assert profile['highest_observed_per_property']['anatomical/probe_ba']['checkpoint'] == 'encoder_b'
    assert 'percentage points' in html


def test_report_has_only_download_links_and_escapes_user_text(tmp_path):
    cfg = configuration(dataset='<script>alert("x")</script>')
    rows = measurements(cfg)
    add_evidence(tmp_path, rows)
    leaf = tmp_path / 'seeds/seed_42/encoder_a--cls'
    features = leaf / 'features'
    features.mkdir()
    (features / 'controlled features.npz').write_bytes(b'example')
    (leaf / 'controlled_embedding_bundle.npz').write_bytes(b'example')
    (tmp_path / 'run.json').write_text('{}')
    profile = write_report(tmp_path, cfg, rows)
    html = (tmp_path / 'report.html').read_text()
    links = Tables(html).links
    assert set(links) == {'run.json', 'scores.csv', 'profile.json'}
    assert '<summary>Artifacts</summary>' not in html
    assert any(item['path'] == 'seeds/seed_42/encoder_a--cls/acquisition.json'
               for item in profile['evidence'])
    assert '<script>alert(' not in html
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in html
    assert 'src="http' not in html and 'href="http' not in html


def test_explicit_artifact_path_supports_module_reports(tmp_path):
    cfg = configuration()
    rows = measurements(cfg)
    output = tmp_path / 'combined'
    artifacts = tmp_path / 'acquisition'
    add_evidence(artifacts, rows)
    for row in rows:
        row['path'] = str(artifacts / 'seeds' / f'seed_{row["seed"]}' / f'{row["checkpoint"]}--{row["feature"]}' / 'acquisition.json')
    profile = write_report(output, cfg, rows)
    html = (output / 'report.html').read_text()
    assert '69 source volumes' in html
    assert any(item['path'] == '../acquisition/seeds/seed_42/encoder_a--cls/acquisition.json'
               for item in profile['evidence'])


def test_registered_alias_matches_canonical_rows(tmp_path):
    from reform3d.core.checkpoints import resolve_checkpoint_config
    cfg = configuration(checkpoints=['Med3DINO-REL-96'])
    rows = measurements(cfg)
    for row in rows:
        row['checkpoint'] = resolve_checkpoint_config(cfg['checkpoints'][0]).checkpoint_key
    profile = write_report(tmp_path, cfg, rows)
    assert profile['status'] == 'complete'
    assert profile['metrics'][0]['checkpoint'] == 'med3dino_rel_96'


def test_browser_sorting_handles_directions_ties_and_top_k_metrics():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed to execute report browser interactions')
    harness = r'''
const assert = require('assert').strict;
const vm = require('vm');
function row(name, transfer, drift, top5) {
  const attributes = {'data-transfer': transfer, 'data-drift': drift, 'data-top-5': top5};
  return {dataset: {checkpoint: name}, rank: {textContent: ''},
    getAttribute(key) { return attributes[key]; }, querySelector() { return this.rank; }};
}
const body = {rows: [row('c', .3, .4, .9), row('b', .8, .2, .8), row('a', .1, .2, .7)],
  appendChild(row) { this.rows.splice(this.rows.indexOf(row), 1); this.rows.push(row); }};
const table = {tBodies: [body], caption: {textContent: ''}};
const select = {dataset: {table: 'test'}, addEventListener(event, handler) { this.handler = handler; }};
const control = {hidden: true, querySelector() { return select; }};
const document = {querySelectorAll() { return [control]; }, getElementById() { return table; }};
vm.runInNewContext(SCRIPT, {document});
assert.equal(control.hidden, false);
function sort(metric, direction) {
  select.value = metric;
  select.selectedOptions = [{dataset: {direction}, textContent: metric}];
  select.handler();
  return body.rows.map(row => row.dataset.checkpoint);
}
assert.deepEqual(sort('transfer', 'desc'), ['b', 'c', 'a']);
assert.deepEqual(sort('drift', 'asc'), ['a', 'b', 'c']);
assert.deepEqual(body.rows.map(row => row.rank.textContent), [1, 1, 3]);
assert.deepEqual(sort('top@5', 'desc'), ['c', 'b', 'a']);
assert.ok(table.caption.textContent.includes('top@5'));
'''
    result = subprocess.run([node, '-e', 'const SCRIPT = ' + json.dumps(SCRIPT) + ';\n' + harness],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
