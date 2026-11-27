import numpy as np

coordinates = "spherical"

# Physical model parameters
Rin = 1.0
Zeff = 1.2
YZ2 = 1.0


# Analytical flow profiles

def density(r, theta, phi):
    return np.divide(
        1.0,
        r**2,
        out=np.zeros_like(r),
        where=(r >= Rin)
    )


def temperature(r, theta, phi):
    return np.ones_like(r)


def electron_per_baryon(r, theta, phi):
    return np.full_like(r, YZ2 / Zeff)


def ion_charge2_per_baryon(r, theta, phi):
    return np.full_like(r, YZ2)


# Opaque objects

opaque_objects = [
    {
        "shape": "sphere",
        "center": (0, 0, 0),
        "radius": 1.0
    },
    {
        "shape": "cylinder",
        "center": (0, 2, 0),
        "radius": 0.8,
        "height": 0.5,
        "axis": (0, 0, 1)
    }
]