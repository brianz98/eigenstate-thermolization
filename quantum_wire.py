#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import pickle
%matplotlib widget

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
        # photon_energies goes from -N to +N as required, cavity_modes is order like [0,1,-1,2,-2,3,-3,..,N,-N] (doubly degenerate other than lowest mode)
        self.photon_energies = np.array([self.e_q(_) for _ in range(-self.n,self.n+1)])
        self.cavity_modes = np.array([(_,_) for _ in self.photon_energies[self.n+1:]]).flatten()
        self.cavity_modes = np.insert(self.cavity_modes,0,self.photon_energies[self.n])
        
        self.mol_energies = np.array([self.rng.normal(self.em_mean, self.em_sigma) for _ in range(self.num_mol)])
        
        self.hamil = np.diag(np.concatenate([self.photon_energies,self.mol_energies])) # Fill in the diagonal elements
        self.hamil = self.hamil.astype('complex128')
        
        self.x_loc = np.array([self.a*_ + self.a*self.rng.uniform(-self.f,self.f) for _ in range(self.num_mol)])
        
        # Fill in the off-diagonal elements
        # We just fill in the light-matter block (first quadrant), and take the complex conjugate for the third quadrant.
        for j in range(self.num_mol,2*self.num_mol):
            # This is the matter loop, we put it outside so mu_j only needs to be drawn once
            mu_j = self.rng.normal(1, self.mu_sigma)
            for q in range(self.num_mol):
                # omega_q was already computed in _diag
                omega_q = self.photon_energies[q]
                self.hamil[q,j] = (self.omega_r/2)*np.sqrt(self.mol_energies[j-self.num_mol]/(self.num_mol*omega_q))*(mu_j)*np.exp(-((q*2*np.pi/(self.a*self.num_mol))*(self.x_loc[j-self.num_mol])-0.5*np.pi)*1j)
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
        # Bin the LP eigenvalues first
        weights,bins = np.histogram(self.lp_eigvals,bins=num_bins)
        bins = (bins[:-1]+bins[1:])/2
        idx = np.cumsum(weights) # get the index ranges to add up the invidual LDOS in the range
        idx = np.insert(idx,0,0)
        ldos = np.array([np.sum(np.conjugate(self.eigvecs[self.num_mol:,_])*self.eigvecs[self.num_mol:,_])/self.num_mol for _ in range(self.num_mol)])

        binned_ldos = np.array([np.sum(ldos[idx[i]:idx[i+1]]) for i in range(len(idx)-1)])
        
        f,ax = plt.subplots(figsize=(8,8))
        ax.bar(bins,binned_ldos,width=(bins[1]-bins[0]))
        
    def refresh_rng(self):
        self.rng = np.random.default_rng()
        
    def get_ipr(self, window_width=50):
        """
        Inverse participation ratio
        """
        pi = 0.0
        for psi in range(len(self.eigvals)):
            for n in range(self.num_mol+self.n-window_width,self.num_mol+self.n+window_width):
                pi += (np.conjugate(self.eigvecs[n,psi])*self.eigvecs[n,psi])**2

        pi /= window_width*2
        pi = pi.real
        
        return pi
    
    def get_r_index(self, window_width=50):
        """
        Gets the r index as defined in eqn. 10 of 10.1016/j.aop.2021.168469. 
        0.5307 is the theoretical GOE result, and Poisson is 0.3863.
        """
        lvls = self.lp_eigvals[self.n-window_width:self.n+window_width]
        s = lvls[1:]-lvls[:-1]
        ra = np.array([min(s[i],s[i+1])/max(s[i],s[i+1]) for i in range(len(s)-1)])
        ra = np.average(ra)
        
        return ra
    
    def get_shannon_entropy(self, window_width=50):
        s = 0.0
        for psi in range(len(self.eigvals)):
            pm_psi = np.dot(np.conjugate(self.eigvecs[self.num_mol:,psi]),self.eigvecs[self.num_mol:,psi])
            pm_psi = pm_psi.real
            for n in range(self.num_mol+self.n-window_width,self.num_mol+self.n+window_width):
                pi = np.conjugate(self.eigvecs[n,psi])*self.eigvecs[n,psi]
                pi = pi.real
                s -= pi/pm_psi * np.log2(pi/pm_psi)

        s /= len(qw.eigvals)
        
        return s
        
def unpickle_wire(fname):
    with open(fname, 'br') as fhandle:
        qw = pickle.load(fhandle)
    
    return qw

def get_properties(qw, nreal, fname=None, window_width=50):
    w = np.zeros(nreal,3)
    for r in range(nreal):
        qw.refresh_rng()
        qw.generate_hamil()
        qw.diag_hamil()
        w[r,0] = qw.get_r_index(window_width)
        w[r,1] = qw.get_ipr(window_width)
        w[r,2] = qw.get_shannon_entropy(window_width)

    #avg = np.array([np.average(w[:i+1]) for i in range(len(w))])
    #err = np.array([avg[i]/np.sqrt((i+1)*window_width*2) for i in range(nreal)])
    
    #if fname is not None:
    #    np.savetxt(f'data/{fname}',avg)
    
    #if plot:
    #    f,ax = plt.subplots(figsize=(8,8))
    #    ax.errorbar(np.arange(1,21),avg,yerr=err,marker='x')
    
    return np.average(w, axis=0)

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

def get_ldos(qw, nreal, num_bins=50, start=50, end=50, hist_range=(0,0.002)):
    eigvals = np.zeros((nreal, qw.num_mol))
    ldos = np.zeros((nreal, qw.num_mol))
    
    for i in range(nreal):
        qw.generate_hamil()
        qw.diag_hamil()
        eigvals[i,:] = qw.lp_eigvals[:]
        ldos[i,:] = np.array([np.sum(np.conjugate(qw.eigvecs[qw.num_mol:,_])*qw.eigvecs[qw.num_mol:,_])/qw.num_mol for _ in range(qw.num_mol)])
        qw.refresh_rng()

    
    # Bin the LP eigenvalues first
    eigvals = eigvals.flatten()
    sort_indx  = np.argsort(eigvals)
    ldos_sorted = (ldos.flatten())[sort_indx]
    weights,bins = np.histogram(eigvals,bins=num_bins)
    bins = (bins[:-1]+bins[1:])/2
    idx = np.cumsum(weights) # get the index ranges to add up the invidual LDOS in the range
    idx = np.insert(idx,0,0)

    binned_ldos = np.array([np.sum(ldos_sorted[idx[i]:idx[i+1]]) for i in range(len(idx)-1)])

    f,ax = plt.subplots(figsize=(8,8))
    ax.bar(bins,binned_ldos,width=(bins[1]-bins[0]))
