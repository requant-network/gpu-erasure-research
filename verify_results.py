"""Verify recorded experiment sources without executing GPU code."""
import hashlib,json,pathlib
count=0
for path in sorted(pathlib.Path('results').glob('*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    for name,expected in record.get('source_sha256',{}).items():
        assert not pathlib.Path(name).is_absolute() and ':' not in name
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected,(path,name)
        count+=1
    for name,expected in record.get('result_sha256',{}).items():
        assert hashlib.sha256((pathlib.Path('results')/name).read_bytes()).hexdigest()==expected,(path,name)
assert count
print(count,'recorded source/result hashes verified')
