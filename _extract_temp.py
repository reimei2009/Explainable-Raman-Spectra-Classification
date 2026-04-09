import os, sys
from pathlib import Path

ROOT = Path(r'C:\Users\ADMIN\Desktop\Prj\KLTN')

# Exec bootstrap.py with __name__ != '__main__' so main() is not called
with open(ROOT / 'bootstrap.py', 'r', encoding='utf-8') as f:
    source = f.read()

namespace = {'__name__': 'bootstrap_extract', '__file__': str(ROOT / 'bootstrap.py')}
exec(compile(source, 'bootstrap.py', 'exec'), namespace)

FILES = namespace['FILES']
print(f'Found {len(FILES)} files in FILES dict:')
for k in FILES:
    print(f'  {k}')

for rel_path, content in FILES.items():
    abs_path = ROOT / rel_path.replace('/', os.sep)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abs_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'[OK] Created: {rel_path}')

for d in ['outputs/figures', 'outputs/processed', 'outputs/reports', 'logs', 'notebooks']:
    dir_path = ROOT / d.replace('/', os.sep)
    dir_path.mkdir(parents=True, exist_ok=True)
    gk = dir_path / '.gitkeep'
    gk.touch()
    print(f'[OK] Created: {d}/.gitkeep')

print('Done.')
