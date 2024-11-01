import numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

import os
import shutil
import time

# ------------------------------------------------------------------------------
# Preamble - copies this script to unique directory for each time it is run.
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
# ------------------------------------------------------------------------------
# Simulation Code
# Solves incompressible NS eqs for Cavity lid problem using projection method with staggered grid.
# ------------------------------------------------------------------------------

def interpolate(u, v):
    u_avg = (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:])/4
    v_avg = (v[:-1, :-1] + v[1:, :-1] + v[1:, 1:] + v[:-1, 1:])/4

    # # Stacking arrays is very very slow. If you really want to make
    # # arrays have a different shape, do this instead...
    # u_avg = np.zeros_like(v) # shape [n+1, n+2]
    # v_avg = np.zeros_like(u) # shape [n+2, n+1]
    # # the u_avg above is shape [n+1, n], 'missing' two columns, so...
    # u_avg[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:])/4
    # # the v_avg above is shape [n, n+1], 'missing' two rows, so...
    # v_avg[1:-1, :] = (v[:-1, :-1] + v[1:, :-1] + v[1:, 1:] + v[:-1, 1:])/4

    return u_avg, v_avg


def centers_and_corners(u, v):
    u_mid = 0.5*(u[:, 1:] + u[:, :-1])
    v_mid = 0.5*(v[1:, :] + v[:-1, :])

    u_cor = 0.5*(u[1:, :] + u[:-1, :])
    v_cor = 0.5*(v[:, 1:] + v[:, :-1])

    # # Stacking arrays is very very slow. If you really want to make
    # # arrays have a different shape, do this instead...
    # u_avg = np.zeros_like(v) # shape [n+1, n+2]
    # v_avg = np.zeros_like(u) # shape [n+2, n+1]
    # # the u_avg above is shape [n+1, n], 'missing' two columns, so...
    # u_avg[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:])/4
    # # the v_avg above is shape [n, n+1], 'missing' two rows, so...
    # v_avg[1:-1, :] = (v[:-1, :-1] + v[1:, :-1] + v[1:, 1:] + v[:-1, 1:])/4

    return u_mid, v_mid, u_cor, v_cor


def dudx (u, dx):
    # (u_i+1 - u_i-1)/(2*dx).
    return (u[1:-1, 2:] - u[1:-1, :-2])/(2*dx)


def dudy (u, dy):
    # (u_j+1 - u_j-1)/(2*dy).
    return (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)


def dpdx(p, dx):
    # (p_i+1 - p_i)/dx.
    return (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx


def dpdy(p, dy):
    # (p_j+1 - p_j)/dy.
    return (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy


def d2udx2 (u, dx):
    # (u_i+1 - 2u_i + u_i-1)/(dx**2)
    return (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])/(dx**2)


def d2udy2 (u,dy):
    # (u_j+1 - 2u_j + u_j-1)/(dy**2)
    return (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])/(dy**2)


def set_BCs(u, v, u_in = 7.0):
    # Velocity BCs. 
    # Top wall. Non-slip BC (zero velocities).
    #v[-1, :] = 0 # v=0
    #u[-1, :] = -u[-2, :] # u=0
    # Top wall. ZeroGradient BCs (neumann)
    v[-1, :] = v[-2, :] # dv/dy=0 at half cell from boundary. Backward difference..  
    u[-1, :] = u[-2, :] # du/dy=0 at boundary. 
    # Bottom wall. Non-slip BC (zero velocities).
    #u[0, :] = -u[1, :] # u=0
    #v[0, :] = 0 # v=0
    # Bottom wall. ZeroGradient BC (neumann)
    u[0, :] = u[1, :] # du/dy=0 at boundary.
    v[0, :] = v[1, :] # dv/dy=0 at half cell from boundary. Forward difference.. 
    # Inlet. 
    u[:, 0] = u_in # To set u=u_in. 
    #u[:, 0] = u[:, 1] # du/dx=0
    #v[:, 0] = v[:, 1] # dv/dx=0
    v[:, 0] = -v[:, 1] # v=0
    # Outlet.
    u[:, -1] = u[:, -2] # du/dx = 0 at half cell from boundary. Backward difference..  
    v[:, -1] = v[:, -2] # dv/dx = 0 at outlet. 

    return u, v


def advect_u(u, v_avg, p, dx, dy, dt, nu, rho):
    # u.shape = [n+2, n+1], BUT...
    # u[0, :] and u[-1, :] are GHOST points, while
    # u[1:-1, 0] and u[1:-1, -1] are BOUNDARY points
    # therefore, only u[1:-1, 1:-1] with shape [n, n-1] gets updated.

    # v.shape = [n+1, n+2] while v_avg.shape = [n, n+1], BUT...
    # v_avg[:, 0] and v_avg[:, -1] are BOUNDARY points, same as u pts
    # so only v_avg[:, 1:-1] are located at u pts that get updated.
    un = u.copy()

    u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
                    - un[1:-1, 1:-1] * dudx(un, dx)
                    - v_avg[:, 1:-1] * dudy(un, dy)
                    + nu * (d2udx2(un, dx) + d2udy2(un, dy))
                    )
    return u


def advect_u_alt(u, v, p, dx, dy, dt, nu, rho):
    u_mid = 0.5*(u[:, 1:] + u[:, :-1]) # u at cell centers
    u_cor = 0.5*(u[1:, :] + u[:-1, :]) # u at cell corners
    v_cor = 0.5*(v[:, 1:] + v[:, :-1]) # v at cell corners

    duudx = (u_mid[1:-1, 1:]**2 - u_mid[1:-1, :-1]**2)/dx
    duvdy = (u_cor[1:, 1:-1]*v_cor[1:, 1:-1] - u_cor[:-1, 1:-1]*v_cor[:-1, 1:-1])/dy
    Laplu = nu * (d2udx2(u, dx) + d2udy2(u, dy))

    u[1:-1, 1:-1] += dt * (Laplu - duudx - duvdy)

    return u


def correct_u(u, p, dx, dt, rho):
    u[1:-1, 1:-1] -= dpdx(p, dx) * rho * dt
    return u


def advect_v (v, u_avg, p, dx, dy, dt, nu, rho):
    # v.shape = [n+2, n+1], BUT...
    # v[:, 0] and v[:, -1] are GHOST points, while
    # v[0, 1:-1] and v[-1, 1:-1] are BOUNDARY points
    # therefore, only v[1:-1, 1:-1] with shape [n-1, n] gets updated.

    # u.shape = [n+2, n+1] while u_avg.shape = [n+1, n], BUT...
    # u_avg[0, :] and u_avg[-1, :] are BOUNDARY points, same as v pts
    # so only u_avg[1:-1, :] are located at v pts that get updated.
    vn = v.copy()

    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
                    - u_avg[1:-1, :] * dudx(vn, dx)
                    - vn[1:-1, 1:-1] * dudy(vn, dy)
                    + nu * (d2udx2(vn, dx) + d2udy2(vn, dy))
                    )
    return v


def advect_v_alt(v, u, p, dx, dy, dt, nu, rho):
    v_mid = 0.5*(v[1:, :] + v[:-1, :]) # v at cell centers
    u_cor = 0.5*(u[1:, :] + u[:-1, :]) # u at cell corners
    v_cor = 0.5*(v[:, 1:] + v[:, :-1]) # v at cell corners

    dvvdy = (v_mid[1:, 1:-1]**2 - v_mid[:-1, 1:-1]**2)/dy
    duvdx = (u_cor[1:-1, 1:]*v_cor[1:-1, 1:] - u_cor[1:-1, :-1]*v_cor[1:-1, :-1])/dx
    Laplv = nu * (d2udx2(v, dx) + d2udy2(v, dy))

    v[1:-1, 1:-1] +=  dt * (Laplv - dvvdy - duvdx)
    return v


def correct_v(v, p, dy, dt, rho):
    v[1:-1, 1:-1] -= dpdy(p, dy) * rho * dt
    return v


def unpslit_euler_b(u, v, u_avg, v_avg, dx, dy, dt):
    # Central difference (1 cell).
    du_dx = (u[1:-1, 1:] - u[1:-1, :-1]) / dx # (u_j+1,i+1-u_j+1,i)/dx, central difference (1cell)
    dv_dy = (v[1:, 1:-1] - v[:-1, 1:-1]) / dy #  v_j+1,i+1-v_j,i+1/dy, central difference (1cell)
    du_dy = (u_avg[1:, :] - u_avg[:-1, :]) / dy # (u_avg_j+1,i-u_avg_j,i)/dy, central difference (1cell)
    dv_dx = (v_avg[:, 1:] - v_avg[:, :-1]) / dx # (v_avg_j,i+1-v_avg_j,i)/dx, central difference (1cell)
    b = (du_dx + dv_dy)/dt - du_dx**2 - 2*du_dy*dv_dx - dv_dy**2

    # Colin introduced an error by having b get multiplied by rho twice,
    # once here, once in pressure_poison. He deleted the extra rho here.
    # Since rho = 1, this never actually showed up as a mistake.

    return b


def split_chorin_b(u, v, dx, dy, dt):
    # Central difference (1 cell).
    du_dx = (u[1:-1, 1:] - u[1:-1, :-1]) / dx
    dv_dy = (v[1:, 1:-1] - v[:-1, 1:-1]) / dy

    # return divergence divided by dt
    return (du_dx + dv_dy) / dt


def pressure_poisson(p, b, dx, dy, tol, maxiter):
    """Solve the Poisson equation using Jabobi's method.
    """
    err = np.inf # Initialize huge error.
    nit = 0 # Reset num iterations. 
    pcoef = 0.5 / (dx**2 + dy**2) # Simplifies code 
    b *= dx**2 * dy**2 / (2*(dx**2 + dy**2))

    while err > tol and nit < maxiter:
        pn = p.copy()

        p[1:-1, 1:-1] = (pcoef * ((pn[1:-1, 2:] + pn[1:-1, :-2])*dy**2
                         + (pn[2:, 1:-1] + pn[:-2, 1:-1])*dx**2) - b)

        # BCs
        p_in=4
        #p[:, 0] = 2*p_in-p[:, 1]  # p=p_in=4 at x=0
        p[:, 0] = p[:, 1] # dp/dx=0 at x=0. 
        p[:, -1] = -p[:, -2] # p = 0 at x = L.
        #p[:, -1] = p[:, -2] # dp/dx = 0 at x = L.
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0.
        p[-1, :] = p[-2, :] # dp/dx = 0 at y = 2.

        # EXPLANATION:
        # There are no p pts at the BOUNDARY,
        # only GHOST points on other side, so...
        # p[-1] is at location y = 2 + dy/2, and
        # p[-2] is at location y = 2 - dy/2
        # thus, if interpolating to exactly y = 2, then we want
        # p_avg = (p[-2] + p[-1])/2 = 0, rearranging, we get
        # p[-1] = -p[-2]

        err = np.mean((p[1:-1, 1:-1] - pn[1:-1, 1:-1])**2)**0.5
        nit += 1

    #print(f'p iters: {nit}')

    return p


def chorin_projection_step(u, v, p, rho, nu, dt, dx, dy, tol, maxiter):
    """1) Advection-Diffusion. 
        2) Solve Poisson. 
         3) Correct velocities (add p term). 
    """
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
    Co = 0.5 # Diffusive courant number
    dt_diff = Co * (1/dx**2 + 1/dy**2)**-1 / (2 * nu)

    while t < tf:
        # Compute time step. 
        # Advective Courant number. 
        u_max = max(np.max(np.abs(u[1:-1, 1:-1])),
                    np.max(np.abs(v[1:-1, 1:-1]))) + 1.0e-20
        dt_adv = cfl * dx / u_max
        dt = min(dt_adv, dt_diff) # Choose conservative. 
        # 
        u, v, p = chorin_projection_step(u, v, p, rho, nu, dt, dx, dy, tol, maxiter)

        if n % 100 == 0:
            print (f'Step: {n}, t = {t:0.3e}, dt = {dt:0.3e}')

        t += dt
        n += 1

    return u, v, p


def plot_divergence(u, v, fig, ax):
    """
    Plots the 2D divergence field given the velocity components u and v.
    dudx and dvdy are calculated using central difference.
    """
    du_dx = (u[1:-1, 1:] - u[1:-1, :-1]) / dx # (u_j+1,i+1-u_j+1,i)/dx, central difference (1cell)
    dv_dy = (v[1:, 1:-1] - v[:-1, 1:-1]) / dy #  v_j+1,i+1-v_j,i+1/dy, central difference (1cell)
    divergence = du_dx + dv_dy

    cp = ax.imshow(divergence, cmap='seismic', origin='lower',
                   aspect='equal')
    fig.colorbar(cp, ax=ax).set_label('Divergence')
    ax.set_title('2D Divergence Field')


def plot_pressure(pressure, fig, ax):
    """
    Plots the 2D pressure field.
    """
    cp = ax.imshow(pressure[1:-1, 1:-1], cmap='plasma', origin='lower',
                   aspect='equal')
    fig.colorbar(cp, ax=ax).set_label('Pressure')
    ax.set_title('2D Pressure Field')


def plot_u(u, fig, ax):
    """
    Plots the 2D x-velocity field.
    """
    # Plot the pressure field
    cp = ax.imshow(u[1:-1, 1:-1], cmap='plasma', origin='lower',
                   aspect='equal')
    fig.colorbar(cp, ax=ax).set_label('U-velocity')
    ax.set_title('2D U-velocity Field')

# Define parameters.
L=6  # Physical length of the domain.
#n = 80 # Number of cell columns/rows.
ny = 30 # Ensure cells are square
nx = L*ny
rho = 1.0 # Density.
nu = 0.1 # Kinematic viscosity.
tf = 20.0 # final simulation time
cfl = 0.1 # CFL number
tol = 1.0e-6  # Poisson solver tolerance threshold value.
maxiter = 100 # Max number of iterations on the Poisson solver.

# Cell lengths. 
#dx = L / (nx - 1) # Horizontal cell length.
dx = L / nx  # Horizontal cell length.
#dy = dx # Vertical cell length.
dy = 1 / ny # Assume vertical length is 1. 
print ('dx= ', dx)
print ('dy= ', dy)
# ICs.
u = numpy.zeros((ny+2, nx+1))
v = numpy.zeros((ny+1, nx+2))
p = np.zeros((ny+2, nx+2)) + 1e-20

# Start the timer
start_time = time.time()
# Run solver
u, v, p = cavity_flow(u, v, p, rho, nu, cfl, tf, dx, dy, tol, maxiter)
# End the timer
end_time = time.time()
# Calculate the duration and print it
duration = end_time - start_time
print(f"Simulation took {duration:.2f} seconds to run.")

# plot div(u), p, u
fig, ax = plt.subplots(1, 3, figsize=(14, 4),
                       sharey=True, sharex=True, dpi=300)
plot_divergence(u, v, fig, ax[0])
plot_pressure(p, fig, ax[1])
plot_u(u, fig, ax[2])

fig.suptitle(f'cells={ny}x{nx}, tol={tol}, maxiter={maxiter}, final time: {tf}s')
fig.savefig(f'{mdir}/triple_panel.png')
