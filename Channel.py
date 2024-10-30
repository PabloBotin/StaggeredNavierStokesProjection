import numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

import os
import shutil
import time

# ------------------------------------------------------------------------------
# Simulation Code
# Solves incompressible NS eqs for Cavity lid problem using projection method with staggered grid.
# ------------------------------------------------------------------------------
def interpolate(u, v):
    u_avg = (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:])/4
    v_avg = (v[:-1, :-1] + v[1:, :-1] + v[1:, 1:] + v[:-1, 1:])/4

    return u_avg, v_avg


def dudx(u, dx):
    # (u_i+1 - u_i-1)/(2*dx).
    return (u[1:-1, 2:] - u[1:-1, :-2])/(2*dx)


def dudy(u, dy):
    # (u_j+1 - u_j-1)/(2*dy).
    return (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)


def d2udx2 (u, dx):
    # (u_i+1 - 2u_i + u_i-1)/(dx**2)
    return (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])/(dx**2)


def d2udy2 (u,dy):
    # (u_j+1 - 2u_j + u_j-1)/(dy**2)
    return (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])/(dy**2)


def set_BCs(u, v):
    ## Horizontal Faces:
    # Note that v points are located at horizontal faces, but u points
    # are not, therefore, u GHOST points naturally enforce BCs,
    # while v would *normally* use BOUNDARY points, unless a Neumann BC
    # is needed!

    # Top Face - ADDED v ghost cells!!
    # v[-1] is GHOST point, v[-2] is BOUNDARY point, v[-3] is INTERIOR point
    v[-1, :] = v[-3, :] # GHOST points
    u[-1, :] = u[-2, :] # GHOST points

    # Bottom Face - ADDED v ghost cells!!
    # v[0] is GHOST point, v[1] is BOUNDARY point, v[2] is INTERIOR point
    v[0, :] = v[2, :] # GHOST points
    u[0, :] = u[1, :] # GHOST points

    ## Vertical Faces:
    # Note that u points are located at vertical faces, but v points are
    # not, therefore, v GHOST points naturally enforce BCs,
    # while u would *normally* use BOUNDARY points, unless a Neumann BC
    # is needed!

    # Left Face - ADDED u ghost cells!!
    # u[0] is GHOST point, u[1] is BOUNDARY point, u[2] is INTERIOR point
    u[:, 0]  = u[:, 2] # GHOST points
    v[:, 0]  = v[:, 1] # GHOST points

    # Right Face - ADDED u ghost cells!!
    # u[-1] is GHOST point, u[-2] is BOUNDARY point, u[-3] is INTERIOR point
    u[:, -1] = u[:, -3] # GHOST points
    v[:, -1] = v[:, -2] # GHOST points

    return u, v


def advect_u(u, v_avg, p, dx, dy, dt, nu, rho):
    # u now has extra ghost points in x. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to v and p is now offset by 1 on each side
    un = u.copy()

    u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
                    - un[1:-1, 1:-1] * dudx(un, dx)
                    - v_avg[1:-1, :] * dudy(un, dy)
                    + nu * (d2udx2(un, dx) + d2udy2(un, dy))
                    )
    return u


def advect_u_alt(u, v, p, dx, dy, dt, nu, rho):
    # u now has extra ghost points in x. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to v and p is now offset by 1 on each side
    u_mid = 0.5*(u[:, 1:] + u[:, :-1]) # u at cell centers
    u_cor = 0.5*(u[1:, :] + u[:-1, :]) # u at cell corners
    v_cor = 0.5*(v[1:-1, 1:] + v[1:-1, :-1]) # v at cell corners

    duudx = (u_mid[1:-1, 1:]**2 - u_mid[1:-1, :-1]**2) / dx
    duvdy = (u_cor[1:, 1:-1] * v_cor[1:, :]
             - u_cor[:-1, 1:-1] * v_cor[:-1, :]) / dy
    Laplu = nu * (d2udx2(u, dx) + d2udy2(u, dy))

    u[1:-1, 1:-1] += dt * (Laplu - duudx - duvdy)

    return u


def correct_u(u, p, dx, dt, rho):
    # u now has extra ghost points in x. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to v and p is now offset by 1 on each side
    u[1:-1, 1:-1] -= (p[1:-1, 1:] - p[1:-1, :-1]) * rho * dt / dx
    return u


def advect_v(v, u_avg, p, dx, dy, dt, nu, rho):
    # v now has extra ghost points in y. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to u and p is now offset by 1 on each side
    vn = v.copy()

    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
                    - u_avg[:, 1:-1] * dudx(vn, dx)
                    - vn[1:-1, 1:-1] * dudy(vn, dy)
                    + nu * (d2udx2(vn, dx) + d2udy2(vn, dy))
                    )
    return v


def advect_v_alt(v, u, p, dx, dy, dt, nu, rho):
    # v now has extra ghost points in y. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to u and p is now offset by 1 on each side
    v_mid = 0.5*(v[1:, :] + v[:-1, :]) # v at cell centers
    u_cor = 0.5*(u[1:, 1:-1] + u[:-1, 1:-1]) # u at cell corners
    v_cor = 0.5*(v[:, 1:] + v[:, :-1]) # v at cell corners

    dvvdy = (v_mid[1:, 1:-1]**2 - v_mid[:-1, 1:-1]**2) / dy
    duvdx = (u_cor[:, 1:] * v_cor[1:-1, 1:]
             - u_cor[:, :-1] * v_cor[1:-1, :-1]) / dx
    Laplv = nu * (d2udx2(v, dx) + d2udy2(v, dy))

    v[1:-1, 1:-1] +=  dt * (Laplv - dvvdy - duvdx)
    return v


def correct_v(v, p, dy, dt, rho):
    # v now has extra ghost points in y. Because I'm using them to set a
    # Neumann BC, we still need to solve for the value at the boundary points.
    # However, the indexing wrt to u and p is now offset by 1 on each side
    v[1:-1, 1:-1] -= (p[1:, 1:-1] - p[:-1, 1:-1]) * rho * dt / dy
    return v


def unpslit_euler_b(u, v, u_avg, v_avg, dx, dy, dt):
    # u and v now have extra ghost points, and index differently relative to
    # each other and p compared to the cavity flow solver
    du_dx = (u[1:-1, 2:-1] - u[1:-1, 1:-2]) / dx
    dv_dy = (v[2:-1, 1:-1] - v[1:-2, 1:-1]) / dy
    du_dy = (u_avg[1:, 1:-1] - u_avg[:-1, 1:-1]) / dy
    dv_dx = (v_avg[1:-1, 1:] - v_avg[1:-1, :-1]) / dx
    b = (du_dx + dv_dy)/dt - du_dx**2 - 2*du_dy*dv_dx - dv_dy**2

    return b


def split_chorin_b(u, v, dx, dy, dt):
    # u and v now have extra ghost points, and index differently relative to
    # each other and p compared to the cavity flow solver
    du_dx = (u[1:-1, 2:-1] - u[1:-1, 1:-2]) / dx
    dv_dy = (v[2:-1, 1:-1] - v[1:-2, 1:-1]) / dy

    # return divergence divided by dt
    return (du_dx + dv_dy) / dt


def pressure_poisson(p, b, dx, dy, tol, maxiter):
    """Solve the Poisson equation using Jabobi's method
    """
    err = np.inf
    nit = 0
    pcoef = 0.5 / (dx**2 + dy**2)
    b *= dx**2 * dy**2 / (2*(dx**2 + dy**2))

    P_in = 1.0
    P_out = 0.0

    while err > tol and nit < maxiter:
        pn = p.copy()

        p[1:-1, 1:-1] = (pcoef * ((pn[1:-1, 2:] + pn[1:-1, :-2])*dy**2
                         + (pn[2:, 1:-1] + pn[:-2, 1:-1])*dx**2) - b)

        # BCs
        p[:, 0] = 2.0*P_in - p[:, 1]   # dp/dx = 0 at x = 0.
        p[:, -1] = 2.0*P_out - p[:, -2] # dp/dx = 0 at x = 2.
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0.
        p[-1, :] = p[-2, :] # dp/dx = 0 at y = 2.

        err = np.mean((p[1:-1, 1:-1] - pn[1:-1, 1:-1])**2)**0.5
        nit += 1

    return p


def chorin_projection_step(u, v, p, rho, nu, dt, dx, dy, tol, maxiter):

    # update BC
    u, v = set_BCs(u, v)

    # use the convective-form u * du/dx...
    # u_avg, v_avg = interpolate(u, v)
    # u = advect_u(u, v_avg, p, dx, dy, dt, nu, rho)
    # v = advect_v(v, u_avg, p, dx, dy, dt, nu, rho)

    # ...or the conservative-form d(u*u)dx
    u = advect_u_alt(u, v, p, dx, dy, dt, nu, rho)
    v = advect_v_alt(v, u, p, dx, dy, dt, nu, rho)

    # update BC with new intermediate velocity
    u, v = set_BCs(u, v)

    # then compute b and p with correct u/v BCs
    b = split_chorin_b(u, v, dx, dy, dt) # Calculate b term.
    p = pressure_poisson(p, b, dx, dy, tol, maxiter) # Solve Poisson eq.

    # then correct the velocity with pressure
    u = correct_u(u, p, dx, dt, rho)
    v = correct_v(v, p, dy, dt, rho)

    return u, v, p


def unsplit_euler_step(u, v, p, rho, nu, dt, dx, dy, tol, maxiter):

    # update BC
    u, v = set_BCs(u, v)

    # then interpolate u/v with updated BCs
    u_avg, v_avg = interpolate(u, v)

    # then compute b and p with correct u/v average
    b = unpslit_euler_b(u, v, u_avg, v_avg, dx, dy, dt) # Calculate b term.
    p = pressure_poisson(p, b, dx, dy, tol, maxiter) # Solve Poisson eq.

    # use the convective-form u * du/dx...
    u = advect_u(u, v_avg, p, dx, dy, dt, nu, rho)
    v = advect_v(v, u_avg, p, dx, dy, dt, nu, rho)

    # ...or the conservative-form d(u*u)dx
    # u = advect_u_alt(u, v, p, dx, dy, dt, nu, rho)
    # v = advect_v_alt(v, u, p, dx, dy, dt, nu, rho)

    # then add the RHS pressure gradient term
    u = correct_u(u, p, dx, dt, rho)
    v = correct_v(v, p, dy, dt, rho)

    return u, v, p


def cavity_flow(u, v, p, rho, nu, cfl, tf, dx, dy, tol, maxiter):
    t = 0.0
    n = 0
    Co = 0.5 # diffusion courant number
    dt_diff = Co * (1/dx**2 + 1/dy**2)**-1 / (2 * nu)

    while t < tf:
        u_max = max(np.max(np.abs(u[1:-1, 1:-1])),
                    np.max(np.abs(v[1:-1, 1:-1]))) + 1.0e-20
        dt_adv = cfl * dx / u_max
        dt = min(dt_adv, dt_diff)

        u, v, p = chorin_projection_step(u, v, p, rho, nu, dt, dx, dy, tol, maxiter)

        if n % 100 == 0:
            print (f'Step: {n}, t = {t:0.3e}, dt = {dt:0.3e}')

        t += dt
        n += 1

    return u, v, p


# ------------------------------------------------------------------------------
# Run Simulation
# ------------------------------------------------------------------------------
# Define parameters.
L = 1.0  # Physical length of the y-domain.
R = 4  # Ratio of x to y dimensions
ny = 20 # Number of cell columns
nx = R*ny # Number of cell rows.
rho = 1.0 # Density.
nu = 0.1 # Kinematic viscosity.
tf = 10.0 # final simulation time
cfl = 0.1 # CFL number
tol = 1.0e-7  # Poisson solver tolerance threshold value.
maxiter = 1e3 # Max number of iterations on the poisson solver.

# Calculations
dx = L*R / nx # Horizontal cell length.
dy = L / ny # Vertical cell length.

# ICs.
u = numpy.zeros((ny+2, nx+3))
v = numpy.zeros((ny+3, nx+2))
p = np.zeros((ny+2, nx+2)) + 1e-20

# Run solver
u, v, p = cavity_flow(u, v, p, rho, nu, cfl, tf, dx, dy, tol, maxiter)


# ------------------------------------------------------------------------------
# Plotting preamble - copies this script to unique directory for each time it is run.
# ------------------------------------------------------------------------------
today = time.strftime('%Y_%m_%d')
run_id = f"run{str(int(time.time()))[-5:]}" # 5 digits of current time in milliseconds
root = os.getcwd()
mdir = f"{root}/figures/{today}/{run_id}"
os.makedirs(mdir, exist_ok=True)
this_file = os.path.basename(__file__)
shutil.copy(this_file, f"{mdir}/{os.path.basename(this_file)}")

rcParams.update({
    ## Constrained Layout and tiny margins are necessary
    'figure.constrained_layout.use': True,
    'axes.xmargin': 0.005,
    'axes.ymargin': 0.005,
    'font.size': 8,
    'axes.formatter.min_exponent': 2,
    'axes.formatter.useoffset': True,
    'axes.formatter.offset_threshold': 6,
    ## These are merely stylistic, feel free to adjust
    'figure.subplot.left': 0.05,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.05,
    'figure.subplot.top': 0.95,
    'figure.subplot.wspace': 0.1,
    'figure.subplot.hspace': 0.1,
    'figure.constrained_layout.h_pad': 0.05,
    'figure.constrained_layout.w_pad': 0.05,
    'figure.constrained_layout.hspace': 0.05,
    'figure.constrained_layout.wspace': 0.05
    })


def plot_divergence(u, v, fig, ax):
    """
    Plots the 2D divergence field given the velocity components u and v.
    dudx and dvdy are calculated using central difference.
    """
    # u and v now have extra ghost points, and index differently relative to
    # each other and p compared to the cavity flow solver
    du_dx = (u[1:-1, 2:-1] - u[1:-1, 1:-2]) / dx
    dv_dy = (v[2:-1, 1:-1] - v[1:-2, 1:-1]) / dy
    divergence = du_dx + dv_dy
    vmax = np.abs(divergence).max()
    cp = ax.imshow(divergence, cmap='seismic', origin='lower',
                   aspect='equal', vmin=-vmax, vmax=vmax)
    fig.colorbar(cp, ax=ax)
    ax.set_title('2D Divergence Field')


def plot_pressure(p, fig, ax):
    """
    Plots the 2D pressure field.
    """
    cp = ax.imshow(p[1:-1, 1:-1], cmap='plasma', origin='lower',
                   aspect='equal')
    fig.colorbar(cp, ax=ax)
    ax.set_title('2D Pressure Field')


def plot_u(u, fig, ax):
    """
    Plots the 2D x-velocity field.
    """
    cp = ax.imshow(u[1:-1, 2:-2], cmap='plasma', origin='lower',
                   aspect='equal')
    fig.colorbar(cp, ax=ax)
    ax.set_title('2D U-velocity Field')

# plot div(u), p, u
fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.0),
                       sharey=True, sharex=True, dpi=600)
plot_divergence(u, v, fig, ax[0])
plot_pressure(p, fig, ax[1])
plot_u(u, fig, ax[2])

fig.suptitle("Open Flow, P_x0 = 1.0, P_x1 = 0.0, BCs are CENTERED zero-gradient."
             f"\ntf={tf:0.1f}, cfl={cfl:0.1f}, tol={tol:0.1e}, maxiter={maxiter:0.1e}")
fig.savefig(f'{mdir}/triple_panel.png')
