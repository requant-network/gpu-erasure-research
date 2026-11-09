"""Exact CPU checks for the GF(256) Cauchy erasure-code contract."""
import argparse,hashlib,itertools,json,pathlib,random
from experiments.field import encode,generator,inverse,multiply,polynomial_product,power,recovery

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results/cpu-study.json');args=parser.parse_args()
    checked=0
    for a in range(256):
        for b in range(256):assert multiply(a,b)==polynomial_product(a,b);checked+=1
    for a in range(1,256):assert multiply(a,power(a,254))==1
    rng=random.Random(20261109);patterns=0
    for k,p in ((2,2),(4,2),(8,4),(16,4)):
        shards=[bytes(rng.randrange(256) for _ in range(17)) for _ in range(k)]
        g=generator(k,p);coded=encode(g,shards);assert coded[:k]==shards
        choices=list(itertools.combinations(range(k+p),p))
        if len(choices)>500:choices=rng.sample(choices,100)
        for missing in choices:
            survivors,coeff=recovery(k,p,missing)
            assert encode(coeff,[coded[i] for i in survivors])==[coded[i] for i in missing]
            patterns+=1
    try:inverse([[1,1],[1,1]])
    except ValueError:pass
    else:raise AssertionError('Singular matrix accepted')
    record=dict(polynomial='0x11d',multiplication_pairs=checked,nonzero_inverses=255,erasure_patterns=patterns,
                scope='Independent polynomial oracle, systematic Cauchy code, exhaustive small erasure patterns and seeded larger patterns; not a DA protocol.',
                source_sha256={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in ('cpu_study.py','experiments/field.py')})
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(checked,'field products;',patterns,'exact erasure recoveries passed')

if __name__=='__main__':main()
