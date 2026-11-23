#!/usr/bin/env python3
"""Reproduce the exactly solvable one-magnon XXZ receiver benchmark.

No fitted transport exponent or unarchived simulation data enter the calculation.
The one-magnon matrix is extracted from the actual many-body Hamiltonian.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import jv, airy
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def xxz_full(L: int, Delta: float, Jxy: float) -> np.ndarray:
    """H = sum (Jxy/2)(S+ S- + S- S+) + Delta Sz Sz, periodic.

    Basis: computational bit 1 denotes Sz=+1/2; bit 0, Sz=-1/2.
    """
    dim=1<<L
    H=np.zeros((dim,dim), dtype=np.float64)
    for x in range(dim):
        for j in range(L):
            k=(j+1)%L
            z_j = .5 if x & (1<<j) else -.5
            z_k = .5 if x & (1<<k) else -.5
            H[x,x]+=Delta*z_j*z_k
            if ((x>>j) & 1) != ((x>>k) & 1):
                y=x ^ (1<<j) ^ (1<<k)
                H[y,x]+=Jxy/2
    return H


def bessel_advantage(r: int, m: int, t: float, Jxy: float=1.0) -> float:
    x=abs(Jxy)*t
    return float(sum(jv(r+q,x)**2 for q in range(m)))


def first_crossing(r: int,m: int,kappa: float,Jxy:float=1.0):
    eta=kappa*r**(-2/3)
    f=lambda x:bessel_advantage(r,m,x,Jxy)-eta
    tmax=(r+7*r**(1/3))/abs(Jxy)
    grid=np.linspace(0,tmax,max(400,int(tmax*7)))
    prev=f(grid[0])
    for i in range(1,len(grid)):
        cur=f(grid[i])
        if prev<0 <= cur:
            return brentq(f,grid[i-1],grid[i],xtol=2e-12)
        prev=cur
    raise RuntimeError(f'No crossing found for r={r}, m={m}, eta={eta}')


def airy_root(m: int, kappa:float):
    # Largest negative stationary point of Ai, in the argument variable.
    s_peak=1.018792971647471/(2**(1/3))
    F=lambda s:m*2**(2/3)*airy(-2**(1/3)*s)[0]**2
    assert 0<kappa<F(s_peak)
    return brentq(lambda s:F(s)-kappa,-10,s_peak,xtol=1e-13)


def write_csv(path,rows,columns):
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=columns)
        w.writeheader();w.writerows(rows)


def run(outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True)
    L=8; Jxy=1.0; r0=2; m=2
    times=np.linspace(0,7,71)
    deltas=[-1.1,0.0,1.0,2.4]
    data=[]
    maxerr=0.0; delta_spread=0.0
    for Delta in deltas:
        full=xxz_full(L,Delta,Jxy)
        omega=(1<<L)-1
        ix=[omega ^ (1<<j) for j in range(L)]
        sub=full[np.ix_(ix,ix)]
        # Unitarity-independent check: exact interaction shift is -Delta.
        bare=(sub-full[omega,omega]*np.eye(L))
        expected=-Delta*np.eye(L)
        for j in range(L):
            expected[j,(j+1)%L]+=Jxy/2
            expected[j,(j-1)%L]+=Jxy/2
        assert np.max(np.abs(bare-expected))<1e-12
        eig,V=eigh(bare)
        # A ring wavefunction has exact Fourier representation; no approximations.
        k=2*np.pi*np.arange(L)/L
        for t in times:
            ampl=V @ (np.exp(-1j*eig*t)*(V.T@np.eye(L)[:,0]))
            adv_ed=np.sum(np.abs(ampl[r0:r0+m])**2)
            # Convolution kernel of the ring one-magnon sector.
            u=np.array([np.mean(np.exp(1j*k*j-1j*Jxy*t*np.cos(k))) for j in range(L)])
            adv_ft=np.sum(np.abs(u[r0:r0+m])**2)
            err=float(abs(adv_ed-adv_ft));maxerr=max(maxerr,err)
            data.append(dict(L=L,Delta=Delta,Jxy=Jxy,source=0,receiver_start=r0,receiver_width=m,t=f'{t:.10f}',adv_ED=f'{adv_ed:.14e}',adv_Fourier=f'{adv_ft:.14e}',abs_error=f'{err:.3e}'))
    write_csv(outdir/'xxz_finite_ring_independent_validation.csv',data,list(data[0]))
    assert maxerr<1e-12, f'XXZ exact diagonalization inconsistent with Fourier: {maxerr}'
    byD=np.array([[float(row['adv_ED']) for row in data if row['Delta']==d] for d in deltas])
    delta_spread=float(np.max(np.ptp(byD,axis=0)))
    assert delta_spread<1e-12, f'Unexpected Delta dependence in one-magnon sector: {delta_spread}'

    plt.figure(figsize=(6.6,3.8))
    for d in deltas:
        sub=[row for row in data if row['Delta']==d]
        plt.plot([float(row['t']) for row in sub],[float(row['adv_ED']) for row in sub],label=fr'$\Delta={d:g}$',lw=1.6)
    plt.xlabel(r'$J_{xy}t$');plt.ylabel(r'Optimal receiver advantage $\mathcal{A}_B(t)$')
    plt.title('Finite-ring XXZ one-magnon identity ($L=8$, B={2,3})')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout();plt.savefig(outdir/'xxz_one_magnon_delta_independence.pdf');plt.close()

    kapp=.14; rs=[24,48,96,192,384,768]
    rows=[]
    plt.figure(figsize=(6.6,3.8))
    for m in [1,2]:
        s_airy=airy_root(m,kapp)
        x=[];y=[]
        for r in rs:
            t=first_crossing(r,m,kapp,Jxy)
            sc=(t-r/Jxy)/(r**(1/3)/Jxy)
            x.append(r**(-1/3));y.append(sc)
            rows.append(dict(distance=r,receiver_width=m,kappa=kapp,Jxy=Jxy,threshold=kapp*r**(-2/3),first_crossing=t,scaled_shift=sc,airy_limit=s_airy,abs_shift_error=abs(sc-s_airy)))
        plt.plot(x,y,'o-',label=fr'$m={m}$; exact Bessel',markersize=4)
        plt.axhline(s_airy,ls='--',lw=1,label=fr'$m={m}$; Airy limit')
    write_csv(outdir/'xxz_exact_airy_first_crossings.csv',rows,list(rows[0]))
    plt.xlabel(r'$r^{-1/3}$')
    plt.ylabel(r'$(|J_{xy}|\tau_{\eta_r}-r)/r^{1/3}$')
    plt.title(r'XXZ one-magnon first crossing ($\eta_r=0.14r^{-2/3}$)')
    plt.legend(fontsize=8,ncol=2);plt.tight_layout()
    plt.savefig(outdir/'xxz_one_magnon_exact_airy_crossings.pdf');plt.close()
    # Assert non-fitted asymptotic convergence to the analytic rising Airy branch.
    for m in [1,2]:
        sub=[row for row in rows if row['receiver_width']==m]
        assert sub[-1]['abs_shift_error']<sub[0]['abs_shift_error']
    record={
       'model':'periodic spin-1/2 XXZ (single down spin above all-up vacuum)',
       'Hamiltonian':'sum_j (Jxy/2)(S+_j S-_(j+1)+h.c.) + Delta Sz_j Sz_(j+1)',
       'state':'vacuum vs one local spin flip at site 0',
       'finite_ring_L':L,'independent_Delta_checks':deltas,
       'max_absolute_ED_vs_Fourier_error':maxerr,
       'max_absolute_Delta_signal_variation':delta_spread,
       'infinite_chain_asymptotic_distances':rs,'kappa':kapp,
       'first_crossing_definition':'first physical time of crossing; scanned earliest positive root',
       'no_fitting':'asymptotic Airy constants fixed before evaluating Bessel first crossings',
       'numerical_libraries':{'numpy':np.__version__},
    }
    (outdir/'validation_summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
    for m in [1,2]:
        sub=[row for row in rows if row['receiver_width']==m]
        print('m=',m,'Airy s=',sub[0]['airy_limit'],'largest-distance shift=',sub[-1]['scaled_shift'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,default=Path(__file__).parent)
    run(ap.parse_args().outdir)
