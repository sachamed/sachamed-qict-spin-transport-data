from pathlib import Path
import json
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.special import iv

# Verify against direct 2^L-state SSEP generator; no use of formula in constructing L.
L=6; D=0.7; a=0; p=0.4; pm=0.17; pp=0.68
n=1<<L
states=np.array(list(itertools.product([0,1],repeat=L)),dtype=int)
lookup={tuple(row):i for i,row in enumerate(states)}
G=np.zeros((n,n))
for j, s in enumerate(states):
    for b in range(L):
        c=(b+1)%L
        if s[b]!=s[c]:
            new=s.copy(); new[b],new[c]=new[c],new[b]
            G[lookup[tuple(new)],j]+=D
            G[j,j]-=D
P0=np.prod(np.where(states==1,p,1-p),axis=1)
q_plus=np.prod(np.where(states==1,p,1-p),axis=1)
q_minus=q_plus.copy()
for x in range(n):
    q_plus[x]*=(pp if states[x,a] else 1-pp)/(p if states[x,a] else 1-p)
    q_minus[x]*=(pm if states[x,a] else 1-pm)/(p if states[x,a] else 1-p)
assert abs(sum(q_plus)-1)<1e-12
# single discrepancy kernel on ring
W=np.zeros((L,L))
for b in range(L):
    for c in [(b-1)%L,(b+1)%L]: W[c,b]+=D; W[b,b]-=D

def marginal(q,B):
    ans=np.zeros(1<<len(B))
    for i,s in enumerate(states):
        k=sum(int(s[b])<<j for j,b in enumerate(B))
        ans[k]+=q[i]
    return ans

rows=[]
for t in [0,0.1,0.9,2.3,4.2]:
    M=expm(t*G); qdiff=M@(q_plus-q_minus)
    P=expm(t*W)[:,a]
    for B in [(1,),(2,3),(0,4),(1,2,3),tuple(range(L))]:
        exact=0.5*np.sum(np.abs(marginal(qdiff,B)))
        # same formula by evaluating an independent receiver Bernoulli product average
        w=P[list(B)]
        prod=[]
        for cfg in itertools.product([0,1],repeat=len(B)):
            prob=np.prod([p if z else 1-p for z in cfg])
            prod.append(prob*abs(np.dot(w,np.array(cfg)-p)))
        predicted=abs(pp-pm)*sum(prod)/(2*p*(1-p))
        rows.append({'t':t,'B':list(B),'tv_direct':exact,'tv_formula':predicted,'error':abs(exact-predicted)})
errors=[r['error'] for r in rows]
# Additional exact 2-site enhancement at half filling
p2=0.5; t=1.3; w=expm(t*W)[:,a]; B=(2,3)
counter=sum((0.5**2)*abs(np.dot(w[list(B)],np.array(cfg)-p2)) for cfg in itertools.product([0,1],repeat=2))
formula=abs(pp-pm)/(2*p2*(1-p2))*counter
maxsite=abs(pp-pm)*max(w[list(B)])
result={'model':'ring SSEP, L=6, D=0.7, p=0.4','n_comparisons':len(rows),'max_abs_error':max(errors),'two_site_p_half':formula,'max_one_site':maxsite,'two_site_error':abs(formula-maxsite),'rows':rows}
with open(Path(__file__).resolve().parent/'block_formula_validation.json','w') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
assert max(errors)<1e-12
assert abs(formula-maxsite)<1e-12
