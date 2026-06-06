import numpy as np

def pressure(U, Speed):

    Cp = 1 - Speed**2/U**2

    return Cp

def rgeometry(AoA):
    xc, yc = np.loadtxt("points.txt", unpack=True)

    xs = xc * np.cos(AoA) + yc * np.sin(AoA)
    ys = -xc * np.sin(AoA) + yc * np.cos(AoA)

    return xs, ys
