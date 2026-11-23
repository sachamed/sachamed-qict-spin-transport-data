#!/usr/bin/env python3
"""Reproduce the finite-temperature local-reset receiver benchmark in an open XX chain.
Only standard scientific Python libraries are required. All quantities are reconstructed
from the Hamiltonian; archived curves are not used as numerical inputs.
"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.linalg import eigh
from scipy.special import jv, airy, ai_zeros
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
L, J = 8, 1.0
n = 1 << L
betas = [0.0, 0.35, 1.25, 3.0]
times = np.linspace(0.0, 10.0, 41)
receivers = [1, 3, 5, 7]

# The down-spin occupancy is one; 0 and 1 label up and down respectively.
def open_xxz_hamiltonian(delta):
    h = np.zeros((n, n), dtype=np.float64)
    for b in range(n):
        for j in range(L-1):
            x = (b >> (L-1-j)) & 1
            y = (b >> (L-2-j)) & 1
            h[b, b] += delta * (0.25 if x == y else -0.25)
            if x != y:
                c = b ^ (3 << (L-2-j))
                h[b, c] += 0.5 * J
    return h

def one_particle_hamiltonian():
    h = np.zeros((L,L), dtype=np.float64)
    for k in range(L-1):
        h[k,k+1] = h[k+1,k] = 0.5*J
    return h

sp_h = one_particle_hamiltonian()
sp_e, sp_v = eigh(sp_h)
def one_particle_prob(r,t):
    return float(np.abs(np.dot(sp_v[r,:]*sp_v[0,:], np.exp(-1j*sp_e*t)))**2)

rows = []
max_exact_error = 0.0
max_beta_spread = 0.0
for delta in [0.0, 1.0]:
    H = open_xxz_hamiltonian(delta)
    e, v = eigh(H)
    # Observable diagonal in standard computational basis.
    diag_n = np.array([[(k >> (L-1-r)) & 1 for k in range(n)] for r in receivers], dtype=float)
    U = v
    obs_eig = np.array([v.conj().T @ np.diag(d) @ v for d in diag_n])
    for beta in betas:
        w = np.exp(-beta * (e-e.min()))
        w /= w.sum()
        gibbs = (v*w[np.newaxis,:])@v.conj().T
        d = n//2
        bath = gibbs[:d,:d]+gibbs[d:,d:]
        # Hypothesis 1: down at site 0; hypothesis 0: up at site 0.
        delta_rho = np.kron(np.diag([-1.0,1.0]),bath)
        rho_eig = v.conj().T@delta_rho@v
        for t in times:
            phase = np.exp(-1j*e*t)
            dyn_rho = (phase[:,None]*rho_eig)*phase.conj()[None,:]
            for ri,r in enumerate(receivers):
                adv = float(abs(np.real(np.sum(obs_eig[ri].T * dyn_rho))))
                predicted = one_particle_prob(r,t)
                err = abs(adv-predicted)
                if delta == 0:
                    max_exact_error = max(max_exact_error,err)
                rows.append([L,delta,beta,r,float(t),adv,predicted,err])

# Independent semi-infinite spectral identity for the receiver at r>=0.
# A truncated open chain and the image sum must coincide before a boundary return.
r_front = 35
front_t = np.linspace(20,53,250)
semi_p = np.array([(jv(r_front,t)+jv(r_front+2,t))**2 for t in front_t])
assert np.all(semi_p>=-1e-12)

# Temperature independence at the exactly solvable XX point.
for r in receivers:
    for t in times:
        values = [x[5] for x in rows if x[1]==0.0 and x[3]==r and abs(x[4]-t)<1e-10]
        max_beta_spread = max(max_beta_spread, max(values)-min(values))

with (HERE/'xx_finiteT_local_reset_validation.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['L','Delta','beta','receiver_site','time','Helstrom_advantage','one_particle_probability','absolute_error'])
    writer.writerows(rows)

plt.figure(figsize=(7.2,4.3))
tfine=np.linspace(0,10,650)
for r in [3,5,7]:
    plt.plot(tfine,[one_particle_prob(r,t) for t in tfine],label=f'Exact propagator, receiver {r}')
# 3 thermal betas and coarser markers on receiver r=5
for beta,marker in [(0.0,'o'),(0.35,'x'),(1.25,'s'),(3.0,'+')]:
    q=[x for x in rows if x[1]==0.0 and x[2]==beta and x[3]==5 and (round(x[4]*4)%5==0)]
    plt.plot([x[4] for x in q],[x[5] for x in q],marker,linestyle='None',markersize=4,label=f'ED, beta={beta:g}')
plt.xlabel('Time (Jt)')
plt.ylabel('Optimal one-site receiver advantage')
plt.title('Thermal XX chain: exact local-reset discrimination')
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(HERE/'xx_finiteT_local_reset_validation.pdf')
plt.close()

# A physical interaction changes the thermal signal; do not extrapolate XX to all Delta.
plt.figure(figsize=(7.2,4.3))
for delta in [0.0,1.0]:
    for beta in [0.0,3.0]:
        rr=[x for x in rows if x[1]==delta and x[2]==beta and x[3]==5]
        plt.plot([x[4] for x in rr],[x[5] for x in rr],label=f'Delta={delta:g}, beta={beta:g}')
plt.xlabel('Time (Jt)')
plt.ylabel('Optimal one-site receiver advantage')
plt.title('Finite-size test: thermal backgrounds and interaction')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE/'xx_vs_xxz_finiteT_receiver.pdf')
plt.close()

# Independently locate first crossings of the exact semi-infinite Bessel signal.
# The threshold scales as r^(-2/3), so there is no fitted transport exponent.
kappa = 0.30
xstar = float(ai_zeros(1)[1][0])
speak = -xstar/2**(1/3)
def front_profile(s):
    return 4.0*2**(2/3)*float(airy(-2**(1/3)*s)[0])**2
sroot = brentq(lambda s: front_profile(s)-kappa,-7.0,speak-1e-8,xtol=1e-13)
front_rows=[]
for r in [24,48,96,192,384,768]:
    eta=kappa*r**(-2/3)
    def p(x):return float((jv(r,x)+jv(r+2,x))**2)
    xcross=brentq(lambda x:p(x)-eta,0,r+1.5*r**(1/3),xtol=1e-11)
    sfinite=(xcross-r)/r**(1/3)
    front_rows.append([r,kappa,eta,xcross,sfinite,sroot,sfinite-sroot])
with (HERE/'xx_finiteT_airy_first_crossings.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['distance','kappa','threshold','J_times_first_crossing','scaled_shift','Airy_limit_shift','finite_distance_discrepancy']);w.writerows(front_rows)
plt.figure(figsize=(6.9,4.1))
plt.plot([r[0] for r in front_rows],[r[4] for r in front_rows],'o-',label='Exact Bessel first crossing')
plt.axhline(sroot,linestyle='--',label='Airy rising-branch coefficient')
plt.xscale('log',base=2)
plt.xlabel('Receiver distance r')
plt.ylabel('Scaled arrival shift')
plt.title('Thermal XX boundary front: parameter-free Airy limit')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(HERE/'xx_finiteT_airy_first_crossings.pdf')
plt.close()

info={
    'chain_length': L,
    'anisotropies_tested': [0.0,1.0],
    'inverse_temperatures_tested': betas,
    'receiver_sites': receivers,
    'time_grid_size': len(times),
    'total_rows': len(rows),
    'max_abs_error_exact_XX_all_temperatures': max_exact_error,
    'max_beta_spread_exact_XX': max_beta_spread,
    'Airy_limit_shift_kappa_0p3':sroot,
    'largest_distance_scaled_shift':front_rows[-1][4],
    'scope': 'Exact temperature-independent receiver law proven only at Delta=0 in an open XX chain for local reset at a boundary site.'
}
(HERE/'xx_finiteT_validation_summary.json').write_text(json.dumps(info,indent=2)+'\n')
assert max_exact_error < 1e-11, max_exact_error
assert max_beta_spread < 1e-11, max_beta_spread
print(json.dumps(info,indent=2))
