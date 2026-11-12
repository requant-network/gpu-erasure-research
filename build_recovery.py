"""Build the recovery-only ISA-L wrapper against the existing pinned library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = 'ab01645a2d9ab581477c703103fe3450c4696845'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('libdir', type=Path)
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    library = args.libdir / 'libisal.so'; assert library.exists()
    Path('build').mkdir(exist_ok=True)
    subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
                    '-I' + str(args.upstream / 'include'), 'native/isal_recovery.cpp',
                    '-L' + str(args.libdir), '-Wl,-rpath,' + str(args.libdir.resolve()),
                    '-lisal', '-o', 'build/libisal_recovery.so'], check=True)
    record = dict(revision=REVISION,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('build_recovery.py', 'native/isal_recovery.cpp')},
                  binary_sha256={'libisal.so': hashlib.sha256(library.read_bytes()).hexdigest(),
                                 'libisal_recovery.so': hashlib.sha256(Path('build/libisal_recovery.so').read_bytes()).hexdigest()})
    Path('results/recovery-build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
