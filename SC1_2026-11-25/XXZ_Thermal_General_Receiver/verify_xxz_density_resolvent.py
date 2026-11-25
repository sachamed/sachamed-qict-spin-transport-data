"""Check finite-ring XXZ density/current resolvent identities and reset-source bounds.
Finite-size floating-point validation only, not a thermodynamic scaling test.
Dependencies: NumPy and SciPy. Run with Python 3.
"""
import numpy as np
from scipy.linalg import eigh
from itertools import product
mx=0.; max_ann=0.; n=0; max_src=0.; rows=[]
for L,delta,beta in product([4,6,8],[1.,1.5],[.3,.7]):
    d=2**L;s=np.arange(d);occ=np.array([(s>>(L-1-x))&1 for x in range(L)])
    H=np.zeros((d,d));J=[]
    for x in range(L):
        y=(x+1)%L;H+=np.diag(delta*(.5-occ[x])*(.5-occ[y]));b=s[occ[x]!=occ[y]];c=b^(1<<(L-1-x))^(1<<(L-1-y));H[c,b]+=.5
        j=np.zeros((d,d),complex);j[c,b]=.5j*(occ[y,b]-occ[y,c]);J.append(j)
    E,V=eigh(H);w=np.exp(-beta*(E-E.min()));w/=w.sum();gap=E[:,None]-E[None,:];weight=w[None,:]/L
    Js=np.array([V.T@j@V for j in J]);rho=(V*w)@V.T;bath=rho[:d//2,:d//2]+rho[d//2:,d//2:];D=V.T@np.kron(np.diag([-1.,1.]),bath)@V
    X=V@(D/w[:,None])@V.T;perm=((s>>1)|((s&1)<<(L-1)));inv=np.argsort(perm);Xs=[X]
    for _ in range(1,L): Xs.append(Xs[-1][np.ix_(inv,inv)])
    Xs=np.array([V.T@x@V for x in Xs])
    for k in 2*np.pi*np.arange(1,L)/L:
        phase=np.exp(-1j*k*np.arange(L));Q=V.T@np.diag(phase@(occ-.5))@V;jk=np.einsum('x,xij->ij',phase,Js);xm=np.einsum('x,xij->ij',phase.conj(),Xs).conj().T
        nq=weight*np.abs(Q)**2;nj=weight*np.abs(jk)**2;chi=nq.sum();a2=abs(1-np.exp(-1j*k))**2
        Y=xm-Q/chi;yn=weight*np.abs(Y)**2;mix=weight*Y.conj()*Q
        max_src=max(max_src,abs(mix.sum()))
        for e in [.025,.1,.3,1.]:
            sig=np.sum(nj*e/(e*e+gap*gap));R=np.sum(nq*e/(e*e+gap*gap));rhs=e*(chi-e*R)/a2
            mx=max(mx,abs(sig-rhs));max_ann=max(max_ann,max(0.,sig-e*chi/a2));n+=1
            # exact mixed spectral bound on each cutoff, using scaled denominator <=1 outside
            for lam in [.25,1.]:
                mask=np.abs(gap)<=lam
                error=abs(np.sum(mix*e/(e-1j*gap)))
                bound=np.sqrt(yn[mask].sum()*chi)+e/lam*np.sqrt(yn.sum()*chi)
                assert error<=bound+1e-10
            rows.append((L,delta,beta,k,e,sig,rhs))
print('configurations',12,'checks',n,'max resolvent difference',mx,'annulus inequality max violation',max_ann,'orthogonal source residual',max_src)

import csv
from pathlib import Path
with Path(__file__).with_suffix('.csv').open('w', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['L','Delta','beta','k','epsilon','conductivity_direct','conductivity_from_density'])
    writer.writerows(rows)
