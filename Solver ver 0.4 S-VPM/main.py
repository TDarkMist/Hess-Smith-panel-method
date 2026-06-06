import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path
import scienceplots
from Geometry import Geometry
from AerodynamicProps import pressure, rgeometry
from Vortex_Panel_solver import vortNormal_influence, XvortVelocity_field, YvortVelocity_field, vortTangential_influence
from Source_Panel_solver import sourceNormal_influence, XsourceVelocity_field, YsourceVelocity_field, sourceTangential_influence

plt.style.use(['science', 'grid', 'dark_background'])
plt.rcParams['text.usetex'] = False  # critical override

U = 10
AoA = 4

_x = np.linspace(-0.5, 1.5, 500)
_y = np.linspace(-0.5, 0.5, 500)

xs,ys = rgeometry(AoA/180 * np.pi)

N = len(xs)-1
Q = int((N)/2)
geom = Geometry(xc=xs, yc=ys)

xlower = np.zeros((Q))
xupper = np.zeros((Q))

for i in range(Q):
    xlower[i] = geom.x_ctr[i]
    xupper[i] = geom.x_ctr[i+Q]


print(xlower)

print(xupper)

I = np.zeros((N,N))
J = np.zeros((N,N))

K = np.zeros((N,N))
L = np.zeros((N,N))

A = np.zeros((N+1, N+1))

Vt = np.zeros((N))

for j in range(N):
    xj, yj = geom.x_start[j], geom.y_start[j]
    phij = geom.theta[j]
    Sj = geom.length[j]
    for i in range(N):
        if i != j:
            xi, yi = geom.x_ctr[i], geom.y_ctr[i]
            phii = geom.theta[i]

            I[i, j] = sourceNormal_influence(Sj, xi, yi, phii, phij, xj, yj)
            J[i, j] = sourceTangential_influence(Sj, xi, yi, phii, phij, xj, yj)

            K[i, j] = -vortNormal_influence(Sj, xi, yi, phii, phij, xj, yj)
            L[i, j] = vortTangential_influence(Sj, xi, yi, phii, phij, xj, yj)

np.fill_diagonal(I, np.pi)
np.fill_diagonal(J, 0)
np.fill_diagonal(K, 0)
np.fill_diagonal(L, 0)

for j in range(N):
    for i in range(N):
        A[i, j] = I[i, j]

for i in range(N):
    A[i, N] = np.sum(K[i, :])

for j in range(N):
    A[N, j] = J[0,j] + J[N-1, j]

A[N, N] = np.sum(L[0, :]) + np.sum(L[N-1, :]) + 2 * np.pi

b = np.zeros((N+1))
for k in range(N):
    b[k] = - U * 2 * np.pi *  np.cos(geom.theta[k] + np.pi/2)

b[N] = - U * 2 * np.pi * (np.sin(geom.theta[0]) + np.sin(geom.theta[N-1]))


lamgam = np.linalg.solve(A, b)

gam = lamgam[N]

print(lamgam)

x,y = np.meshgrid(_x, _y)

u = U + XsourceVelocity_field(N, lamgam[:-1], x, y, geom.x_start[:,None,None], geom.y_start[:,None,None], geom.theta[:,None,None], geom.length[:,None,None]) + XvortVelocity_field(N, gam, x, y, geom.x_start[:,None,None], geom.y_start[:,None,None], geom.theta[:,None,None], geom.length[:,None,None])
v =     YsourceVelocity_field(N, lamgam[:-1], x, y, geom.x_start[:,None,None], geom.y_start[:,None,None], geom.theta[:,None,None], geom.length[:,None,None]) + YvortVelocity_field(N, gam, x, y, geom.x_start[:,None,None], geom.y_start[:,None,None], geom.theta[:,None,None], geom.length[:,None,None])

Tinf = 0

for i in range(N):
    xi, yi = geom.x_ctr[i], geom.y_ctr[i]
    phii = geom.theta[i]

    term1 = U * np.sin(geom.theta[i] + np.pi/2)
    term2 = 0
    term3 = gam/2
    term4 = 0

    for j in range(N):
        if j != i:
            xj, yj = geom.x_start[j], geom.y_start[j]
            phij = geom.theta[j]
            Sj = geom.length[j]

            term2 += lamgam[j]*sourceTangential_influence(Sj, xi, yi, phii, phij, xj, yj)/(2*np.pi)
            term4 += -(gam/(2*np.pi))* vortTangential_influence(Sj, xi, yi, phii, phij, xj, yj)

    Vt[i] = term1 + term2 + term3 + term4

SCp = 1 - Vt**2 / U**2

gamma = gam*np.sum(geom.length)
Cl = 2*gamma/(U)

print("Vt min/max:", Vt.min(), Vt.max())
print("Cp min/max:", SCp.min(), SCp.max())

print(gam)

print(u.shape)
print(v.shape)

xc= np.append(geom.x_end, geom.x_start[0])
yc= np.append(geom.y_end, geom.y_start[0])

speed = np.sqrt(u**2 + v**2)

Cp = pressure(U, speed)

polygon = Path(np.column_stack((geom.x_start, geom.y_start)))
points = np.column_stack((x.ravel(), y.ravel()))
inside = polygon.contains_points(points)

inside = inside.reshape(x.shape)

u = np.ma.array(u, mask=inside)
v = np.ma.array(v, mask=inside)
Cp = np.ma.array(Cp, mask=inside)
speed = np.ma.array(speed, mask=inside)

print("Lift Coefficient", Cl)

fig, axe = plt.subplots(2, 2, figsize=(10,8))
ax = axe[0, 0]
ax.plot(xc, yc)
ax.set_aspect('equal')
ax.set_title('Flow visualization')
ax.streamplot(x, y, u, v, color='white', density=1.5)

ax = axe[0, 1]
ax.plot(xc, yc)
ax.set_aspect('equal')
levels = np.linspace(-3, 1, 80)
norm = colors.TwoSlopeNorm(vmin=Cp.min(),vcenter=0,  vmax=Cp.max())
c = ax.contourf(x, y, Cp, levels=200, norm=norm, cmap='plasma')
ax.set_title('Pressure Coefficient')
fig.colorbar(c, ax=ax, label=r'$C_p$')

ax = axe[1, 0]
ax.plot(xc, yc)
ax.set_aspect('equal')
c = ax.contourf(x, y, speed, levels=200, cmap='viridis')
ax.set_title('Speed')
fig.colorbar(c, ax=ax, label=r'm/s')

ax = axe[1, 1]
ax.invert_yaxis()
ax.plot(xlower, SCp[:Q], color='red', linestyle='dashdot', linewidth=2, label='lower surface')
ax.plot(xupper, SCp[Q:], color='purple', linestyle='dashdot', linewidth=2, label='upper surface')
ax.legend(loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel(r'$C_p$')
plt.show()