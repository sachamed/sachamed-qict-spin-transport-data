"""Finite-ring check of the reset-source expansion and conserved projection.
H = sum (Jxy/2)(S+S-+S-S+) + Jz Sz Sz, Jxy=1, h=0.
Norms are Gibbs susceptibility norms per site. Spectral degeneracies are
identified numerically; these are finite-volume diagnostics, not certificates
of the infinite-volume conserved subspace or hydrodynamic scaling.
"""
import csv
from pathlib import Path
from itertools import product
import numpy as np
from scipy.linalg import eigh

rows=[]
for L,Jz in product([4,6,8],[1.,1.5]):
    d=2**L; s=np.arange(d)
    occ=np.array([(s>>(L-1-x))&1 for x in range(L)])
    q=occ-.5; M=np.diag(q.sum(axis=0)); H=np.zeros((d,d))
    for x in range(L):
        y=(x+1)%L; H+=np.diag(Jz*q[x]*q[y]); b=s[occ[x]!=occ[y]]
        c=b^(1<<(L-1-x))^(1<<(L-1-y)); H[c,b]+=.5
    E,V=eigh(H); m=V.T@M@V
    mask=np.abs(E[:,None]-E[None,:])<1e-10
    # Permutation implementing one lattice translation.
    perm=(s>>1)|((s&1)<<(L-1)); inv=np.argsort(perm)
    for beta in [0.,.025,.05,.1,.2,.5]:
        w=np.exp(-beta*(E-E.min())); w/=w.sum()
        rho=(V*w)@V.T; bath=rho[:d//2,:d//2]+rho[d//2:,d//2:]
        D=np.kron(np.diag([-1.,1.]),bath); Dsum=D.copy(); Dcur=D
        for _ in range(1,L):
            Dcur=Dcur[np.ix_(inv,inv)]; Dsum+=Dcur
        A=(V.T@Dsum@V)/w[None,:]
        def ip(a,b): return np.sum(w[None,:]*a.conj()*b)/L
        def norm(a): return np.sqrt(max(0.,ip(a,a).real))
        chi=ip(m,m).real; y=A-m/chi
        pair=ip(A,m)
        expansion=norm(A-(4+2*beta*Jz)*m)
        resid=norm(y); conserved=norm(mask*y)
        # Energy and magnetization products provide an odd conserved diagnostic.
        # This product is not assumed to be an extensive pseudolocal charge.
        mh=m*(E[None,:]-np.dot(w,E)); mh-=m*ip(m,mh)/chi
        ov=abs(ip(y,mh))/norm(mh) if norm(mh)>1e-12 else 0.
        assert abs(pair-1)<1e-11
        assert conserved<=resid+1e-11
        assert ov<=conserved+1e-10
        rows.append([L,Jz,beta,chi,abs(pair-1),expansion,resid,conserved,ov])
        if beta in [.025,.05,.5]: print(L,Jz,beta,'residual',resid,'projected',conserved,'residual/beta^2',resid/beta**2)
with Path(__file__).with_suffix('.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['L','Jz','beta','chi','unit_pair_error','first_order_expansion_error','source_residual_norm','conserved_residual_norm_finite_ring','overlap_normalized_MH']);w.writerows(rows)
print('rows',len(rows),'max pairing error',max(r[4] for r in rows))
