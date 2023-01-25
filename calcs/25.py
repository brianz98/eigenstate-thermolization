import os
import sys
sys.path.append(os.path.relpath("../"))
from quantum_wire import *

num_mol = 1001
a = 50
em_mean = 1.8
# using log scale
sigma_lo = -5
sigma_hi = 1.5
sigma_points = 200

es = 10**(np.linspace(sigma_lo,sigma_hi,sigma_points))
avgs = np.zeros((len(es),3))

for i, sigma in enumerate(es):
    print(f'sigma={sigma}')
    qw = QuantumWire(num_mol=num_mol, omega_r=0.01, a=a*1e-9, Ly=400e-9, Lz=200e-9, eps=3, em_mean=em_mean, em_sigma=sigma, mu_sigma=0, f=0)
    avg = get_properties(qw, 30, window_width=25)
    print(avg)
    avgs[i,:] = avg

np.save(f'prod-{int(num_mol)}-{int(a)}nm-{em_mean:.2f}-{sigma_lo:.3f}-{sigma_hi:.2f}-{int(sigma_points)}',avgs)
