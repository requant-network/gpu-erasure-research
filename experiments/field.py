"""Independent GF(256), systematic Cauchy code and erasure recovery."""
POLYNOMIAL=0x11d

def multiply(a,b):
    assert 0<=a<256 and 0<=b<256
    out=0
    while b:
        if b&1:out^=a
        b>>=1;a<<=1
        if a&256:a^=POLYNOMIAL
    return out

def polynomial_product(a,b):
    out=0
    for bit in range(8):
        if b>>bit&1:out^=a<<bit
    for bit in range(14,7,-1):
        if out>>bit&1:out^=POLYNOMIAL<<(bit-8)
    return out

def power(a,n):
    out=1
    while n:
        if n&1:out=multiply(out,a)
        a=multiply(a,a);n>>=1
    return out

def inverse(matrix):
    n=len(matrix);assert n and all(len(row)==n for row in matrix)
    work=[list(row)+[int(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=next((i for i in range(column,n) if work[i][column]),None)
        if pivot is None:raise ValueError('Singular code submatrix')
        work[column],work[pivot]=work[pivot],work[column]
        factor=power(work[column][column],254)
        work[column]=[multiply(x,factor) for x in work[column]]
        for i in range(n):
            if i!=column:
                factor=work[i][column]
                work[i]=[a^multiply(factor,b) for a,b in zip(work[i],work[column])]
    return [row[n:] for row in work]

def product(a,b):
    assert a and b and all(len(row)==len(b) for row in a)
    return [[_dot(row,[r[j] for r in b]) for j in range(len(b[0]))] for row in a]

def _dot(a,b):
    result=0
    for x,y in zip(a,b):result^=multiply(x,y)
    return result

def generator(k,parity):
    assert k>0 and parity>0 and k+parity<=256
    return [[int(i==j) if i<k else power(i^j,254) for j in range(k)] for i in range(k+parity)]

def recovery(k,parity,missing):
    assert len(set(missing))==len(missing)<=parity
    assert all(0<=i<k+parity for i in missing)
    g=generator(k,parity);survivors=[i for i in range(k+parity) if i not in missing][:k]
    return survivors,product([g[i] for i in missing],inverse([g[i] for i in survivors]))

def encode(matrix,shards):
    assert shards and len({len(s) for s in shards})==1
    assert all(len(row)==len(shards) for row in matrix)
    size=len(shards[0]);out=[]
    for row in matrix:
        result=bytearray(size)
        for coefficient,shard in zip(row,shards):
            if coefficient:
                for i,x in enumerate(shard):result[i]^=multiply(coefficient,x)
        out.append(bytes(result))
    return out
