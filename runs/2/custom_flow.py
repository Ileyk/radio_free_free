import numpy as np

def density(R, phi, z):
    return np.where(R<0.8,1,0)

def temperature(R, phi, z):
    return 1.0

def electron_per_baryon(R, phi, z):
    return 1.0

def ion_charge2_per_baryon(R, phi, z):
    return 1.0