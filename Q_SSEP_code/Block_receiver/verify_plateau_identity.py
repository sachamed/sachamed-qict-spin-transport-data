"""Check the closed half-filled receiver plateau and arbitrary-filling limit."""
from math import comb, sqrt, pi


def exact_g(m):
    return sum(comb(m, j) * abs(m - 2*j) for j in range(m+1)) / 2**m


def closed_g(m):
    n=(m+1)//2
    return (2*n) * comb(2*n, n) / 4**n


def general_abs_binomial(m,p):
    # Stable recurrence of binomial masses, without factorial overflow.
    q=(1-p)**m
    sm=abs(0-m*p)*q
    for k in range(m):
        q*= (m-k)/(k+1)*p/(1-p)
        sm+=abs(k+1-m*p)*q
    return sm


if __name__=='__main__':
    errs=[abs(exact_g(m)-closed_g(m)) for m in range(1,21)]
    assert max(errs) < 1e-12
    p=.37
    v=general_abs_binomial(500,p)
    expected=sqrt(2*500*p*(1-p)/pi)
    rel=abs(v/expected-1)
    assert rel < .025
    print({'m_values':20,'max_closed_form_error':max(errs),'m_general':500,'p_general':p,'relative_CLT_error':rel})
