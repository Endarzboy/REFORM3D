"""Notebook entry cells must work without an inherited PYTHONPATH or old wheel."""
import json
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / 'notebooks').glob('*.ipynb'))


def setup_source(workflow='pca'):
    name = '01_attention.ipynb' if workflow == 'similarity' else '00_case_selection.ipynb'
    nb = json.loads((ROOT / 'notebooks' / name).read_text())
    return ''.join(next(c for c in nb['cells'] if c['cell_type'] == 'code')['source'])


def test_notebooks_select_their_workflow_from_one_shared_config():
    assert len(NOTEBOOKS) == 3
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['cell_type'] == 'code')['source'])
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'setup']
        assert len(calls) == 1
        workflow = 'similarity' if path.name == '01_attention.ipynb' else 'pca'
        assert {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {'workflow': workflow}


@pytest.mark.parametrize('name,module,entrypoint', [
    ('00_case_selection.ipynb', 'visualization_utils', 'prepare_inputs'),
    ('01_attention.ipynb', 'reform3d.full_volume_features', 'load_or_extract_full_volume_features'),
    ('02_pca.ipynb', 'visualization_utils', 'extract_features'),
])
def test_notebooks_call_their_separate_input_workflows(name, module, entrypoint):
    nb = json.loads((ROOT / 'notebooks' / name).read_text())
    source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(source)
    helper_imports = {alias.name for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module == module
                      for alias in node.names}
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert entrypoint in helper_imports and entrypoint in calls
    assert not calls.intersection({'prepare_case', 'collect_maps', 'plot_gallery', 'select_roi_views'})


@pytest.mark.parametrize('name,plotter', [('01_attention.ipynb', 'plot_full_volume_similarity'),
                                       ('02_pca.ipynb', 'plot_features')])
def test_plotting_cells_pass_receipt_and_explicit_controls(name, plotter):
    nb = json.loads((ROOT / 'notebooks' / name).read_text())
    source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == plotter]
    assert calls
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert len(call.args) >= 3 or 'receipt' in keywords
        assert 'num_slices' in keywords
        if plotter == 'plot_full_volume_similarity':
            assert 'case_id' in keywords
            assert 'mode' not in keywords
        else:
            assert {'mode', 'kind'} <= keywords


def local_config(tmp_path):
    image = tmp_path / 'image.nii.gz'; image.write_bytes(b'input')
    mask = tmp_path / 'mask.nii.gz'; mask.write_bytes(b'annotation')
    weight = tmp_path / 'teacher.pth'; weight.write_bytes(b'weight')
    cases = [{'modality': m, 'image': str(image), 'mask': str(mask)} for m in ['CT', 'MR']]
    cfg = {'checkpoints': {f'med3dino_{p}_96': str(weight) for p in ['iso', 'rel', 'sa']},
           'device': 'cpu', 'pca': {'cases': cases, 'output_dir': 'crop'},
           'similarity': {'cases': [{'id': c['modality'], **c} for c in cases],
                          'output_dir': 'full'}}
    path = tmp_path / 'visualization.json'; path.write_text(json.dumps(cfg))
    return path


def run_setup(tmp_path, cwd, config, *, workflow='pca', root_hint=None, prefix=''):
    diagnostics = "\nprint(sys.modules['reform3d.visualization'].__file__)\nprint([c['modality'] for c in config['cases']])\n"
    script = tmp_path / 'setup.py'; script.write_text(prefix + '\n' + setup_source(workflow) + diagnostics)
    env = {k:v for k,v in os.environ.items() if k not in ['PYTHONPATH','REFORM3D_REPO_ROOT','REFORM3D_VIZ_CONFIG']}
    env.update(REFORM3D_VIZ_CONFIG=str(config), MPLCONFIGDIR=str(tmp_path / 'mpl'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if root_hint:env['REFORM3D_REPO_ROOT']=str(root_hint)
    return subprocess.run([sys.executable,'-I',str(script)],cwd=cwd,env=env,capture_output=True,text=True,timeout=60)


@pytest.mark.parametrize('location', ['repository','notebooks','unrelated'])
@pytest.mark.parametrize('workflow', ['pca', 'similarity'])
def test_setup_resolves_checkout_without_pythonpath(tmp_path, location, workflow):
    cwd={'repository':ROOT,'notebooks':ROOT/'notebooks','unrelated':tmp_path}[location]
    result=run_setup(tmp_path,cwd,local_config(tmp_path),workflow=workflow,
                     root_hint=ROOT if location=='unrelated' else None)
    assert result.returncode == 0,result.stderr
    assert str(ROOT/'src/reform3d/visualization.py') in result.stdout
    assert "['CT', 'MR']" in result.stdout


def test_setup_reports_stale_import_instead_of_mixing_packages(tmp_path):
    prefix="import sys, types\nold=types.ModuleType('reform3d')\nold.__file__='/old/site-packages/reform3d/__init__.py'\nsys.modules['reform3d']=old"
    result=run_setup(tmp_path,ROOT,local_config(tmp_path),prefix=prefix)
    assert result.returncode != 0 and 'Restart the notebook kernel' in result.stderr


def test_setup_explains_missing_config(tmp_path):
    result=run_setup(tmp_path,ROOT,tmp_path/'missing.json')
    assert result.returncode != 0 and 'Copy notebooks/visualization_config.example.json' in result.stderr


def test_setup_names_missing_input(tmp_path):
    config=local_config(tmp_path)
    (tmp_path/'image.nii.gz').unlink()
    result=run_setup(tmp_path,ROOT,config)
    assert result.returncode != 0 and 'CT image:' in result.stderr and 'Missing input files' in result.stderr


def test_attention_notebook_uses_whole_volume_primary_and_additional_cases():
    import ast
    notebook = json.loads((ROOT/'notebooks/01_attention.ipynb').read_text())
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)]
    assert not any(n.func.id in {'plot_head_groups', 'plot_features', 'plot_attention_heads',
                                'plot_reference_similarity', 'extract_features', 'load_features'} for n in calls)
    plots = [n for n in calls if n.func.id == 'plot_full_volume_similarity']
    assert len(plots) == 2
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and any(plot is child for plot in plots for child in ast.walk(node))]
    assert len(loops) == 2
    controls = {}
    for loop in loops:
        cases = tuple(ast.literal_eval(loop.iter))
        plot = next(node for node in ast.walk(loop) if node in plots)
        keywords = {k.arg: k.value for k in plot.keywords}
        assert [arg.id for arg in plot.args] == ['config', 'arrays', 'receipt']
        assert isinstance(keywords['case_id'], ast.Name) and keywords['case_id'].id == 'case_id'
        assert isinstance(keywords['num_slices'], ast.Name)
        controls[cases] = keywords['num_slices'].id
    assert controls == {('MR', 'CT'): 'num_slices',
                        ('MR-additional', 'CT-additional'): 'additional_num_slices'}
