import os
import sys
sys.path.append(os.path.relpath("../"))
from quantum_wire import *

es = np.linspace(1e-3,2.5,50)
avgs = np.zeros(len(es))

for i, sigma in enumerate(es):
    print(f'sigma={sigma}')
    qw = QuantumWire(num_mol=501, omega_r=0.01, a=50e-9, Ly=400e-9, Lz=200e-9, eps=3, em_mean=2.2, em_sigma=sigma, mu_sigma=0, f=0)
    avg = get_r_index(qw, 20, window_width=25)
    print(avg)
    avgs[i] = avg

np.save('energy_disorder-0-2.5',avgs)
