#!/usr/bin/env python
# coding: utf-8

# # Single-particle Eigenstate Thermolization in Polariton Hamiltonians. Photonic Wire Model
# **Ribeiro Group Rotation Project A**
# 
# Brian Zhao, 09 Sept 2022

# The following is adapted from Ribeiro, R. F. _Multimode Polariton Effects on Molecular Energy Transport and Spectral Fluctuations_. Commun Chem 2022, 5 (1), 1–10. https://doi.org/10.1038/s42004-022-00660-0.
# 
# ### Model Hamiltonian
# #### Cavity Hamiltonian
# The empty cavity Hamiltonian is given by
# $$
# H_{\text{L}}=\sum_q \epsilon_q a_q^{\dagger}a_q,
# $$
# where
# $$
# \epsilon_q = \frac{\hbar c}{\sqrt{\epsilon}}\sqrt{q_0^2+q^2},
# $$
# where $q_0=\sqrt{(\pi/L_z)^2+(\pi/L_y)^2}$ is a constant, and $q=2\pi m/L$ ($m\in \mathcal{Z}$) are the _cavity modes_.
# 
# #### Matter Hamiltonian
# The Hamiltonian for the molecules are given by
# $$
# H_{\text{M}}=\sum_{i=1}^{N_{\text{M}}}(\epsilon_{\text{M}}+\sigma_i)b_i^+b_i^-,
# $$
# where $b_i^+=|1_i\rangle\langle 0_i|$ and $b_i^-=|0_i\rangle\langle 1_i|$ creates and annihilates an excitation at the $i$-th molecule respectively, and $\sigma_i$ is drawn from a normal distribution with variance $\sigma^2$.
# 
# #### Light-matter Hamiltonian
# Applying the Coulomb gauge in the rotating-wave approximation (ignoring double raising and lowering), we have
# $$
# H_{\text{LM}}=\sum_{j=1}^{N_{\text{M}}}\sum_q\frac{-i\Omega_{\text{R}}}{2}\sqrt{\frac{\epsilon_{\text{M}}}{N_{\text{M}}\epsilon_q}}\frac{\mu_j}{\mu_0}\left(e^{iqx_j}b_j^+a_q-e^{-iqx_j}a_q^{\dagger}b_j^- \right),
# $$
# where $\Omega_{\text{R}}=\mu_0\sqrt{\hbar\omega_0N_{\text{M}}/2\epsilon LL_yL_z}$, and $\mu_j$ is drawn from a normal distribution with variance $\sigma_{\mu}^2$.
# 
# We assume there is only one photon.

# In[1]:


import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'widget')

q_to_ev = cst.c*cst.hbar/cst.e

class QuantumWire:
    
    def __init__(self, num_mol, omega_r, a, Ly, Lz, eps, em_mean, em_sigma, mu_sigma, f):

        self.num_mol = num_mol # Number of molecules
        
        # The restriction of N_c == N_m should be removed in future
        try:
            assert (num_mol-1)%2 == 0
        except AssertionError:
            raise ValueError('num_mol needs to be 2N+1, where N is a positive integer!')
            
        self.n = int((num_mol-1)/2) # Number of *distinct* cavity modes
        self.omega_r = omega_r # Rabi splitting / eV
        self.a = a # Molecular spacing / nm
        self.Ly = Ly # y-length / nm
        self.Lz = Lz # z-length / nm
        self.L = num_mol*a # x-legnth /nm
        self.eps = eps # Polarisability
        self.em_mean = em_mean # Mean molecular excitation energy / eV
        self.em_sigma = em_sigma*omega_r # STD of molecular excitation energy / eV
        self.mu_sigma = mu_sigma # STD of ratio of mu_j/mu_0
        self.f = f # Fractional spread of molecular positions
        
        self.rng = np.random.default_rng() # Initialise our random number generator
        
        self.photon_energies = None
        self.mol_energies = None
        self.hamil = None
        self.eigvals = None
        self.eigvecs = None
        self.lp_eigvals = None
        self.up_eigvals = None
        self.lp_photonic_content = None
        self.up_photonic_content = None
        
    def e_q(self, m):
        return np.sqrt((np.pi/self.Lz)**2 + (np.pi/self.Ly)**2 + (2*np.pi*m/self.L)**2) * q_to_ev/np.sqrt(self.eps)

    def generate_hamil(self):
        self.photon_energies = np.array([self.e_q(_) for _ in range(-self.n,self.n+1)])
        self.cavity_modes = np.array([(_,_) for _ in self.photon_energies[self.n+1:]]).flatten()
        self.cavity_modes = np.insert(self.cavity_modes,0,self.photon_energies[self.n])
        self.mol_energies = np.array([self.rng.normal(self.em_mean, self.em_sigma) for _ in range(self.num_mol)])
        
        self.hamil = np.diag(np.concatenate([self.photon_energies,self.mol_energies])) # Fill in the diagonal elements
        self.hamil = self.hamil.astype('complex128')
        
        # Fill in the off-diagonal elements
        # We just fill in the light-matter block (first quadrant), and take the complex conjugate for the third quadrant.
        for j in range(self.num_mol,2*self.num_mol):
            # This is the matter loop, we put it outside so mu_j only needs to be drawn once
            mu_j = self.rng.normal(1, self.mu_sigma)
            for q in range(self.num_mol):
                # omega_q was already computed in _diag
                omega_q = self.photon_energies[q]
                self.hamil[q,j] = (self.omega_r/2)*np.sqrt(self.mol_energies[j-self.num_mol]/(self.num_mol*omega_q))*(mu_j)*np.exp(-((q*2*np.pi/(self.a*self.num_mol))*(self.a*(j-self.num_mol)+self.a*self.rng.uniform(-self.f,self.f))-0.5*np.pi)*1j)
                self.hamil[j,q] = np.conjugate(self.hamil[q,j])
    
    def diag_hamil(self):
        self.eigvals, self.eigvecs = np.linalg.eigh(self.hamil)
        self.lp_eigvals = self.eigvals[:self.num_mol]
        self.up_eigvals = self.eigvals[self.num_mol:]
        photonic_content = np.array([sum(np.conjugate(self.eigvecs[:self.num_mol,_])*self.eigvecs[:self.num_mol,_]) for _ in range(self.num_mol*2)])
        photonic_content = photonic_content.astype('float64')
        self.lp_photonic_content = photonic_content[:self.num_mol]
        self.up_photonic_content = photonic_content[self.num_mol:]
        
    def pickle_wire(self, fname):
        with open(fname, 'bw') as fhandle:
            pickle.dump(self, fhandle)
        
    def plot_eigenstates(self, xlim, ylim):
        f,ax = plt.subplots(figsize=(8,8))
        im = ax.scatter(np.arange(0,self.num_mol), self.lp_eigvals, c=self.lp_photonic_content*100, cmap='magma', vmin=0, vmax=100)
        ax.scatter(np.arange(0,self.num_mol), self.up_eigvals, c=self.up_photonic_content, cmap='magma')
        ax.plot(np.arange(0,self.num_mol), self.cavity_modes, linestyle='--', color='tab:green', lw=1)
        ax.axhline(y=self.em_mean,linestyle='--', color='black', lw=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Eigenstate index')
        ax.set_ylabel(r'$E$ / eV')

        f.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label='% photonic content')
        
    def plot_molecular_dos(self, num_bins):
        weights,bins = np.histogram(self.lp_eigvals,bins=num_bins)
        bins = (bins[:-1]+bins[1:])/2
        idx = np.cumsum(weights)
        idx = np.insert(idx,0,0)
        ldos = np.array([np.sum(np.conjugate(self.eigvecs[self.num_mol:,_])*self.eigvecs[self.num_mol:,_])/self.num_mol for _ in range(self.num_mol)])

        binned_ldos = np.array([np.sum(ldos[idx[i]:idx[i+1]]) for i in range(len(idx)-1)])
        
        f,ax = plt.subplots(figsize=(8,8))
        ax.bar(bins,binned_ldos,width=(bins[1]-bins[0]))
        
    def refresh_rng(self):
        self.rng = np.random.default_rng()
        
        
def unpickle_wire(fname):
    with open(fname, 'br') as fhandle:
        qw = pickle.load(fhandle)
    
    return qw

def get_r_index(qw, nreal, fname=None, window_width=50, plot=False):
    """
    Gets the r index as defined in eqn. 10 of 10.1016/j.aop.2021.168469. 
    0.5307 is the theoretical GOE result, and Poisson is 0.3863.
    """
    w = np.zeros(nreal)
    for r in range(nreal):
        qw.refresh_rng()
        qw.generate_hamil()
        qw.diag_hamil()
        lvls = qw.lp_eigvals[qw.n-window_width:qw.n+window_width]
        s = lvls[1:]-lvls[:-1]
        ra = np.array([min(s[i],s[i+1])/max(s[i],s[i+1]) for i in range(len(s)-1)])
        w[r] = np.average(ra)

    avg = np.array([np.average(w[:i+1]) for i in range(len(w))])
    err = np.array([avg[i]/np.sqrt((i+1)*window_width*2) for i in range(nreal)])
    
    if fname is not None:
        np.savetxt(f'data/{fname}',avg)
    
    if plot:
        f,ax = plt.subplots(figsize=(8,8))
        ax.errorbar(np.arange(1,21),avg,yerr=err,marker='x')
    
    return avg[-1]

def get_gap_distribution(qw, nreal, num_bins=150, start=50, end=50, hist_range=(0,0.002)):
    nstates = qw.num_mol - start - end
    gaps = np.zeros((nreal,nstates-1))
    
    if end == 0:
        end = None # Make sure negative indexing works for 0, otherwise a[blah:0] returns empty array.
        
    for i in range(nreal):
        qw.generate_hamil()
        qw.diag_hamil()
        gaps[i,:] = qw.lp_eigvals[1+start:-end]-qw.lp_eigvals[start:-end-1]
        qw.refresh_rng()
        
    gaps_new = gaps.flatten()
    weights, bins = np.histogram(gaps_new, bins=num_bins, range=hist_range)
    bins = (bins[:-1]+bins[1:])/2
    f,ax = plt.subplots(figsize=(8,8))
    ax.bar(bins, weights, width=(bins[1]-bins[0]))
    ax.set_xlim(hist_range)


# ### Basic usage / tutorial
# A `QuantumWire` object can be initialised as follows:
# ```python
# qw = QuantumWire(num_mol=1001, omega_r=0.3, a=10e-9, Ly=400e-9, Lz=200e-9, eps=3, em_mean=2.2, em_sigma=0.2, mu_sigma=0.1, f=0.05)
# ```
# where the arguments are 
# - `num_mol`: Number of molecules. Must be $2N+1$, $N\in\mathcal{Z}^+$. The same number of cavity modes are created, with their quantum numbers $m=-N,-N+1,\dots,0,\dots,N-1,N$
# - `omega_r`: The Rabi splitting, in units of eV.
# - `a`: The average spacing between molecules, in units of meters.
# - `Ly`: The y-length, in units of meters.
# - `Lz`: The z-length, in units of meters.
# - `eps`: The permittivity.
# - `em_mean`: The average molecular excitation energy, in units of eV.
# - `em_sigma`: The standard deviation of the excitation energy, in units of the **fraction of** Rabi splitting
# - `mu_sigma`: The standard deviation of $\mu_j/\mu_0$, unitless.
# - `f`: The range of variation in the position of each molecule as a fraction of `a`, unitless.
# 
# The Hamiltonian can be generated and diagonalized by
# ```python
# qw.generate_hamil()
# qw.diag_hamil()
# ```
# And the eigenvalues and eigenvectors can be accessed by `qw.eigvals` and `qw.eigvecs` respectively. The lower and upper polariton eigenvalues and photonic contents can be accessed by `qw.(l/u)p_(eigvecs/photonic_content)`.
# 
# The lower/upper polariton bands can be plotted, together with their photonic contents:

ngrid = 20
nreal = 15
r_indx = np.zeros((ngrid,ngrid))
for i,f in enumerate(np.linspace(0,0.25,ngrid)):
    for j,mu_sigma in enumerate(np.linspace(0,1.5,ngrid)):
        qw = QuantumWire(num_mol=501, omega_r=0.3, a=10e-9, Ly=400e-9, Lz=200e-9, eps=3, em_mean=2.2, em_sigma=0.2, mu_sigma=mu_sigma, f=f)
        r_indx[i,j] = get_r_index(qw, nreal)

np.savetxt('data/501_0.3_2.2_0.2_0-1.5_0-0.25.dat', r_indx)
