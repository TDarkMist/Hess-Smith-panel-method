import numpy as np

def sourceNormal_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj)*np.cos(phij) - (yi - yj)*np.sin(phij)
    B = (xi - xj)**2 + (yi - yj)**2
    C = np.sin(phii - phij)
    D = -(xi - xj)*np.sin(phii) + (yi - yj)*np.cos(phii)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Iij = (C/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((D-A*C)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Iij


def sourceTangential_influence(Sj, xi, yi, phii, phij, xj, yj):

    A = -(xi - xj) * np.cos(phij) - (yi - yj) * np.sin(phij)
    B = (xi - xj) ** 2 + (yi - yj) ** 2
    C = -np.cos(phii - phij)
    D = (xi - xj) * np.cos(phii) + (yi - yj) * np.sin(phii)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Jij = (C / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((D - A * C) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Jij

def X_influence(x, y, xj, yj, phij, Sj):

    A = -(x - xj)*np.cos(phij) - (y - yj)*np.sin(phij)
    B = (x - xj)**2 + (y - yj)**2
    Cx = -np.cos(phij)
    Dx = (x -xj)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Mxp = (Cx/2) * np.log((Sj**2 + 2*A*Sj + B)/B) + ((Dx-A*Cx)/E) * (np.atan2(Sj+A, E) - np.atan2(A,E))

    return Mxp

def Y_influence(x, y, xj, yj, phij, Sj):
    A = -(x - xj) * np.cos(phij) - (y - yj) * np.sin(phij)
    B = (x - xj) ** 2 + (y - yj) ** 2
    Cy = -np.sin(phij)
    Dy = (y - yj)
    E = np.sqrt(np.maximum(B - A**2, 1e-14))

    Myp = (Cy / 2) * np.log((Sj ** 2 + 2 * A * Sj + B) / B) + ((Dy - A * Cy) / E) * (np.atan2(Sj + A, E) - np.atan2(A, E))

    return Myp

def XsourceVelocity_field(N, lam, x, y, xj, yj, phij,Sj):

    u = np.zeros_like(x)

    u = np.sum((lam[:, None, None]/(2*np.pi)) * X_influence(x, y, xj, yj, phij, Sj), axis=0)

    return u

def YsourceVelocity_field(N, lam, x, y, xj, yj, phij,Sj):

    v = np.zeros_like(y)

    v = np.sum((lam[:, None, None]/(2*np.pi)) * Y_influence(x, y, xj, yj, phij, Sj), axis=0)

    return v





