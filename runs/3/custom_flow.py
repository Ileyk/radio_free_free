import numpy as np

R0=0.8
th0=np.radians(45)  # semi-opening angle of the flow

def density(R, phi, z):
    return np.where(R<R0+np.abs(z)*np.tan(th0),1/(1+(np.abs(z)/R0)*np.tan(th0))**2,0)

def temperature(R, phi, z):
    return 1.0

def electron_per_baryon(R, phi, z):
    return 1.0

def ion_charge2_per_baryon(R, phi, z):
    return 1.0