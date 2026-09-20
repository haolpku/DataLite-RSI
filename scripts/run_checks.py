#!/usr/bin/env python3
"""Run all offline test directories in isolated processes to avoid import collisions."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = {ROOT/'tests'}
    for tree in ('benchmarks', 'evaluation', 'rsi/methods'):
        folders.update(p.parent for p in (ROOT/tree).rglob('test_*.py'))
    failed = []
    for folder in sorted(folders):
        print(f'\nChecking {folder.relative_to(ROOT)}', flush=True)
        run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', str(folder)], cwd=ROOT)
        if run.returncode:
            failed.append(str(folder.relative_to(ROOT)))
    if failed:
        print('Failed test directories: '+', '.join(failed))
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
