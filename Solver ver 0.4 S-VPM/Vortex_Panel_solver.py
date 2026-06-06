import numpy as np

def vortNormal_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj)*np.cos(phij) - (yi - yj)*np.sin(phij)
    B = (xi - xj)**2 + (yi - yj)**2
    C = -np.cos(phii - phij)
    D = (xi - xj)*np.cos(phii) + (yi - yj)*np.sin(phii)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Kij = (C/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((D-A*C)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Kij


def vortTangential_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj) * np.cos(phij) - (yi - yj) * np.sin(phij)
    B = (xi - xj)**2 + (yi - yj)**2
    C = np.sin(phii - phij)
    D = (xi - xj) * np.sin(phii) - (yi - yj) * np.cos(phii)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Lij = (C / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((D - A * C) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Lij

def X_influence(x, y, xj, yj, phij, Sj):

    A = -(x - xj)*np.cos(phij) - (y - yj)*np.sin(phij)
    B = (x - xj)**2 + (y - yj)**2
    Cx = np.sin(phij)
    Dx = -(y - yj)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Nxp = (Cx/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((Dx-A*Cx)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Nxp

def Y_influence(x, y, xj, yj, phij, Sj):
    A = -(x - xj) * np.cos(phij) - (y - yj) * np.sin(phij)
    B = (x - xj) ** 2 + (y - yj) ** 2
    Cy = -np.cos(phij)
    Dy = (x - xj)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Nyp = (Cy / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((Dy - A * Cy) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Nyp

def XvortVelocity_field(N, gam, x, y, xj, yj, phij,Sj):

    u = np.zeros_like(x)

    u = np.sum((-gam/(2*np.pi)) * X_influence(x, y, xj, yj, phij, Sj), axis=0)


    return u

def YvortVelocity_field(N, gam, x, y, xj, yj, phij,Sj):

    v = np.zeros_like(y)

    v = np.sum((-gam / (2 * np.pi)) * Y_influence(x, y, xj, yj, phij, Sj), axis=0)
    return v





