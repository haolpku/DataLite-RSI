"""Release checks: actual CLI entrypoints and generated-source consistency."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('site_builder', ROOT/'scripts/build_site.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReleaseTests(unittest.TestCase):
    def test_all_evaluator_clis(self):
        for name in ('opsd-math-competition-suite', 'math-sft-transfer-suite', 'image-edit-transfer-suite', 'videomme-sft-transfer'):
            with self.subTest(suite=name):
                folder = ROOT/'benchmarks'/name
                flag = [] if name == 'videomme-sft-transfer' else ['--predictions']
                run = subprocess.run([sys.executable,str(folder/'evaluator.py'),*flag,str(folder/'tests/fixture_predictions.jsonl')],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                metrics = json.loads(run.stdout)
                # The video CLI keeps its public output shape; this is the explicit
                # mapping used when assembling a result-manifest phase.
                if name == 'videomme-sft-transfer':
                    exported = {'primary_score':metrics['overall'],**metrics['categories']}
                    self.assertIn('ocr',exported)
                else:
                    exported = metrics
                self.assertGreaterEqual(exported['primary_score'],0)
                self.assertLessEqual(exported['primary_score'],1)

    def test_site_scores_are_derived_from_manifests(self):
        data,items,_ = builder.make_data()
        self.assertEqual(len(data['all']),len(items))
        self.assertEqual(len(data['videos']),5)
        for name,metric,shown,key in data['all']:
            r = json.loads((ROOT/f'results/submissions/{key}/result.json').read_text())
            scale,digits = (1,5) if r['track']=='generative' else (100,2)
            expected = ' → '.join(f"{r['metrics'][phase]['primary_score']*scale:.{digits}f}" for phase in ('baseline','final'))
        self.assertEqual(shown,expected)
        self.assertAlmostEqual(data['math']['opsd']['rows'][1][1],60.3)
        evolver_rows = data['math']['evolver']['rows']
        self.assertEqual(evolver_rows[1][0], 'DataFlow Math-3K · GPT-4o rewrite')
        self.assertEqual(evolver_rows[1][2], 'control')
        self.assertGreater(evolver_rows[-1][1], evolver_rows[1][1])
        self.assertGreater(evolver_rows[2][1], evolver_rows[-1][1])
        self.assertEqual(data['math']['evolver']['gain'], '+1.08 pp')
        self.assertIn('Eight-set',data['math']['evolver']['metric'])
        self.assertIn('Seven-set',data['math']['self']['metric'])

    def test_readme_and_static_page_are_complete(self):
        class Page(HTMLParser):
            def __init__(self):
                super().__init__(); self.ids=[]; self.links=[]
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if 'id' in attrs:self.ids.append(attrs['id'])
                if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
        with tempfile.TemporaryDirectory() as tmp:
            builder.build(output=Path(tmp),revision='a'*40,check=True)
            text=(Path(tmp)/'index.html').read_text()
            page=Page();page.feed(text)
            self.assertEqual(len(page.ids),len(set(page.ids)))
            self.assertNotIn('{{',text)
            self.assertNotIn('window.openai',text)
            for link in page.links:
                if link.startswith('#'):self.assertIn(link[1:],page.ids)
            self.assertIn('58.70 → 62.10',text)
            self.assertIn('noscript',text)


if __name__=='__main__':
    unittest.main()
