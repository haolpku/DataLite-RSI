#!/usr/bin/env python3
"""Build the static research site and README result table from registered results."""
from __future__ import annotations

import argparse
import copy
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/haolpku/DataLite-RSI'
START = '<!-- results:start -->'
END = '<!-- results:end -->'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def make_data(root=ROOT):
    content = read_json(root / 'site/content.json')
    data = copy.deepcopy(content)
    records = {p.parent.name: read_json(p) for p in sorted(root.glob('results/submissions/*/result.json'))}
    listed = {item['id'] for item in content['results']}
    if listed != records.keys():
        raise ValueError('site/content.json must list every registered result exactly once')
    if len(listed) != len(content['results']):
        raise ValueError('duplicate result in site/content.json')
    for item in content['results']:
        record = records[item['id']]
        item['record'] = record
        item['path'] = f"results/submissions/{item['id']}/result.json"
    by_id = {x['id']: x for x in content['results']}
    for key, math in data['math'].items():
        record = records[Path(math['source']).parent.name]
        metrics = record['metrics']
        score = lambda arm: metrics[arm]['primary_score'] * 100
        base_label = 'Qwen2.5-7B base' if key == 'evolver' else 'Base model'
        rows = [[base_label, score('baseline'), 'base']]
        if key == 'opsd':
            control = metrics['controls']['opsd_100_steps']['primary_score'] * 100
            pool = record['settings']['training_pool_size']
            rows.append([f'Full-pool OPSD · {pool:,} examples', control, 'control'])
        elif key == 'evolver':
            refs = metrics['reference_systems']
            control = refs['dataflow_instruct_math3k_original']['primary_score'] * 100
            rows.extend([
                ['DataFlow Math-3K · expert-authored pipeline', control, 'control'],
                [
                    'DataFlow Math-3K · GPT-4o rewrite',
                    refs['dataflow_instruct_math3k_gpt4o_rewrite']['primary_score'] * 100,
                    'base',
                ],
            ])
        else:
            control = score('baseline')
        if key in ('opsd', 'self'):
            for index, row in enumerate(metrics['per_iteration'].values(), 1):
                label = f"Data-lite · {row['selected_samples']:,} examples" if key == 'opsd' else f"Iteration {index} · {row['selected_samples']:,} rows"
                rows.append([label, row['primary_score'] * 100, 'ours'])
        else:
            rows.append(['DataFlow-Evolver', score('final'), 'ours'])
        math['rows'] = rows
        math['gain'] = f"{score('final') - control:+.2f} pp"
    data['images'] = []
    data['videos'] = []
    data['all'] = []
    native = []
    for item in content['results']:
        r = item['record']
        m = r['metrics']
        normalized = r['track'] == 'generative'
        scale, digits = (1, 5) if normalized else (100, 2)
        shown = f"{m['baseline']['primary_score'] * scale:.{digits}f} → {m['final']['primary_score'] * scale:.{digits}f}"
        data['all'].append([item['label'], item['metric'], shown, item['id']])
        if normalized:
            control = m['controls']['stratified_pipeline']
            data['images'].append({'title': r['model']['id'].split('/')[-1], 'rows': [['Base backbone', m['baseline']['primary_score'], 'base'], ['Stratified synthesis', control['primary_score'], 'control'], ['Policy-evolving synthesis', m['final']['primary_score'], 'ours']], 'gain': f"{m['final']['primary_score'] - control['primary_score']:+.5f} vs stratified"})
            for label, arm in [('stratified', control), ('evolving', m['final'])]:
                native.append([r['model']['id'].split('/')[-1] + ' / ' + label, f"{arm['gedit_bench']:.3f}", f"{arm['imgedit_bench']:.3f}"])
        elif r['method_id'] == 'video-rsi':
            data['videos'].append([r['model']['id'].split('/')[-1], m['baseline']['primary_score']*100, m['final']['primary_score']*100])
    data['counts'] = {key: len(list(root.glob(pattern))) for key, pattern in {'method':'rsi/methods/*/method.json','result':'results/submissions/*/result.json','benchmark':'benchmarks/*/benchmark.json','dataset':'datasets/*/dataset.json'}.items()}
    data['records'] = {x['id']: {'status': x['record']['status'], 'source': x['path'], 'protocol': x['metric']} for x in content['results']}
    data.pop('results')
    return data, content['results'], native


def result_table(items):
    lines = ['| Method / model | Protocol / metric | Base → final |', '| --- | --- | --- |']
    for item in items:
        r = item['record']
        factor, digits = (1, 5) if r['track'] == 'generative' else (100, 2)
        values = [r['metrics'][phase]['primary_score'] * factor for phase in ('baseline', 'final')]
        lines.append(f"| [{item['label']}]({item['path']}) | {item['metric']} | {values[0]:.{digits}f} → {values[1]:.{digits}f} |")
    return '\n'.join(lines)


def build(root=ROOT, output=None, revision=None, check=False):
    data, items, native = make_data(root)
    table = result_table(items)
    readme = root / 'README.md'
    old = readme.read_text(encoding='utf-8')
    if START not in old or END not in old:
        raise ValueError('README result-table markers are missing')
    new = old.split(START)[0] + START + '\n\n' + table + '\n\n' + END + old.split(END)[1]
    if check and old != new:
        raise ValueError('README result table is stale; run python scripts/build_site.py')
    if not check:
        readme.write_text(new, encoding='utf-8')
    revision = revision or subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('revision must be a full Git commit SHA')
    data['base'] = f'{REPO}/blob/{revision}/'
    data['revision'] = revision
    esc = lambda value: html.escape(str(value), quote=True)
    opsd = data['math']['opsd']
    rows = ''
    for label, value, kind in opsd['rows']:
        shown = f'{value:.2f}%'
        rows += f'<div class="rs-bar-row"><div class="rs-bar-label"><span>{esc(label)}</span><b>{shown}</b></div><div class="rs-bar-track" role="img" aria-label="{esc(label)}: {shown}"><div class="rs-bar-mark rs-{kind}" style="width:{value}%"></div></div></div>'
    all_rows = ''.join(f'<tr><td><a href="{data["base"]}results/submissions/{esc(i)}/result.json">{esc(n)}</a></td><td>{esc(m)}</td><td>{esc(v)}</td></tr>' for n,m,v,i in data['all'])
    op_record = next(x['record'] for x in items if x['id'] == 'opsd-data-lite-qwen3-8b')
    image_record = next(x['record'] for x in items if x['id'] == 'policy-evolving-edit-synthesis-flux2-klein-9b')
    gains = [f-b for _,b,f in data['videos']]
    replacements = {
        'revision': revision, 'short_revision': revision[:7],
        **{k+'_count': str(v) for k,v in data['counts'].items()},
        'opsd_comparison': f"{opsd['rows'][-1][1]:.1f}% vs {opsd['rows'][1][1]:.1f}%",
        'image_comparison': f"{data['images'][0]['rows'][1][1]:.3f} → {data['images'][0]['rows'][2][1]:.3f}",
        'video_gain_range': f'+{min(gains):.2f} to +{max(gains):.2f} pp',
        'opsd_samples': f"{op_record['settings']['selected_samples'][-1]:,}",
        'image_pairs': f"{image_record['settings']['accepted_pairs']:,}",
        'math_bars': rows, 'opsd_gain': opsd['gain'], 'opsd_protocol': esc(opsd['protocol']),
        'opsd_source': data['base']+opsd['source'], 'all_rows': all_rows,
        'native_rows': ''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in row)+'</tr>' for row in native),
        **{'method_'+k:esc(data['methods']['opsd'][k]) for k in ('title','description','feedback','budget','available')},
        'method_source':data['base']+data['methods']['opsd']['path'],
    }
    page = (root/'site/index.template.html').read_text(encoding='utf-8')
    for key,value in replacements.items():
        page = page.replace('{{'+key+'}}',value)
    if re.search(r'{{[^}]+}}',page):
        raise ValueError('unresolved site template placeholders')
    output = output or root/'_site'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root/'site/assets', output/'assets', dirs_exist_ok=True)
    (output/'index.html').write_text(page,encoding='utf-8')
    payload = json.dumps(data,ensure_ascii=False,indent=2).replace('<','\\u003c')
    (output/'data.js').write_text('window.DATALITE = '+payload+';\n',encoding='utf-8')
    (output/'results.json').write_text(payload+'\n',encoding='utf-8')
    (output/'.nojekyll').touch()
    (output/'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: https://haolpku.github.io/DataLite-RSI/sitemap.xml\n')
    (output/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://haolpku.github.io/DataLite-RSI/</loc></url></urlset>\n')
    print(f'Built {output} from {len(items)} results at {revision[:7]}')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='fail if README data is stale')
    parser.add_argument('--revision',help='full source commit SHA; default HEAD')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    build(output=args.output, revision=args.revision, check=args.check)
