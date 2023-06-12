import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve1d


"""
Python implementation of:
Data generation for the Aliev-Panfilov model from EP-PINNs




https://github.com/martavarela/EP-PINNs/blob/main/Matlab_code/AlievPanfilov1D_RK.m

"""





def AlPan(V, W, a, D):
    # Parameters

    a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    eps = 0.002
    b = 0.15
    h = 1  # cell length [mm]
    # D = 0.1  # diffusion coefficient [mm^2/AU]

    # del2V = 4*D*np.gradient(np.gradient(V, h), h)
    del2V = 4*D*convolve1d(V, [1, -2, 1], axis=0, mode='nearest')
    # del2V = 4 * D * np.diff(V, 2)/h**2
    # del2V = np.concatenate(([0], del2V, [0]))

    dVdt = (-k*V*(V-a)*(V-1) - W*V) + del2V
    dWdt = (eps + mu1*W/(mu2+V))*(-W-k*V*(V-b-1))

    return dVdt, dWdt


def RK4(V, W, dt, a, D):
    k1_V = AlPan(V, W, a, D)[0]
    k1_W = AlPan(V, W, a, D)[1]

    k2_V = AlPan(V + k1_V*dt/2, W + k1_W*dt/2, a, D)[0]
    k2_W = AlPan(V + k1_V*dt/2, W + k1_W*dt/2, a, D)[1]

    k3_V = AlPan(V + k2_V*dt/2, W + k2_W*dt/2, a, D)[0]
    k3_W = AlPan(V + k2_V*dt/2, W + k2_W*dt/2, a, D)[1]

    k4_V = AlPan(V + k3_V*dt, W + k3_W*dt, a, D)[0]
    k4_W = AlPan(V + k3_V*dt, W + k3_W*dt, a, D)[1]

    V = V + (k1_V + 2*k2_V + 2*k3_V + k4_V)*dt/6
    W = W + (k1_W + 2*k2_W + 2*k3_W + k4_W)*dt/6
    return V, W


def solve(n_cells, BCL, ncyc, dt, stim_dur, Va, a, D):

    # Allow boundary conditions to be set
    X = n_cells + 2  # Number of cells + 2 boundary cells

    stigmeo = np.zeros(X, dtype=bool)
    stigmeo[0:5] = True  # Where the stimulus is applied

    gathert = round(1/dt)  # Number of time steps

    t_end = ncyc*BCL  # End time [AU]

    # Initial conditions V and W
    V = np.zeros(X)
    V[:] = 0.01

    W = np.zeros(X)
    W[:] = 0.01

    V_save = np.zeros((n_cells, int(np.ceil(t_end/dt))))
    W_save = np.zeros((n_cells, int(np.ceil(t_end/dt))))
    print(f"V_save shape: {V_save.shape}")

    stim_count = 0
    it_count = 0
    y = np.vstack((V, W))
    print(f" t_end: {t_end}")
    print(f"X: {X}")

    for t in np.arange(dt, t_end+dt, dt):
        it_count += 1

        if t >= stim_count*BCL and stim_count < ncyc:
            V[stigmeo] = Va

        if t >= BCL*stim_count+stim_dur:
            stim_count += 1

        V, W = RK4(V, W, dt, a, D)
        # print(f"y shape: {y.shape}")

        V[1] = V[0]
        V[-2] = V[-1]
        """ 
        if it_count % gathert == 0:
            # print(f"V shape: {V[1:-2].shape}")
            # print(f"V_save shape: {V_save.shape}")
            V_save[:, round(it_count/gathert)] = V[1:-1]
            W_save[:, round(it_count/gathert)] = W[1:-1]
         """
        V_save[:, it_count-1] = V[1:-1]
        W_save[:, it_count-1] = W[1:-1]

    return V_save, W_save


def update_plot(frame_number, V_save, line, text):
    line.set_data(np.arange(V_save.shape[0]), V_save[:, frame_number-1])
    # Add frame number to plot
    text.set_text(f"Frame: {frame_number}")

    return line, text


def animate(V_save):
    fig, ax = plt.subplots()
    ax.set_ylim([0, V_save.max()*1.1])
    ax.set_xlim([0, V_save.shape[0]])
    ax.set_xlabel("Cell")
    ax.set_ylabel("V")

    line, = ax.plot([], [])
    text = ax.text(0.95, 0.95, "", transform=ax.transAxes,
                   ha="right", va="top")

    return FuncAnimation(fig, update_plot, frames=V_save.shape[1], fargs=(V_save, line, text), blit=True)


def main():

    n_cells = 100  # Number of cells
    BCL = 100  # Basic Cycle Length in [AU]: time between repeated stimuli
    ncyc = 1  # Number of cycles: Number times the cell is stimulated

    dt = 0.005  # Time step [AU]

    stim_dur = 1  # Stimulus duration [AU]
    a=0.01
    D=0.1
    Va = 1.0  # Value of V when the stimulus is applied
    V_save, W_save = solve(n_cells, BCL, ncyc, dt, stim_dur, Va, a, D)
    """ 
    V_save = V_save[::-1, ::-1]
    W_save = W_save[::-1, ::-1]

    V_save = V_save[:70, 5000:13000:40]
    W_save = W_save[:70, 5000:13000:40]
    x=np.arange(0.1,20.1,0.1)
    t=np.arange(1,71,1)
    #.mat file with keys: V, W, x, t
    from scipy.io import savemat
    savemat("./data/data1D.mat", {"V": V_save, "W": W_save, "x": x, "t": t})

    for a in np.linspace(0.002, 0.01, 5):
        for D in np.linspace(0.02, 0.1, 5):
            V_save, W_save = solve(n_cells, BCL, ncyc, dt, stim_dur, Va, a, D)
            V_save = V_save[::-1, ::-1]
            W_save = W_save[::-1, ::-1]

            V_save = V_save[:70, 5000:13000:40]
            W_save = W_save[:70, 5000:13000:40]

            print(f"V_save shape: {V_save.shape}")

            np.save(f"./data_params/V_a{a}_D{D}.npy", V_save)
            np.save(f"./data_params/W_a{a}_D{D}.npy", W_save)

    x = np.arange(n_cells)
    t = np.arange(V_save.shape[1])
    np.save("./data_params/x.npy", x)
    np.save("./data_params/t.npy", t)
    # animation = animate(V_save)
    # plt.show()
    """

if __name__ == "__main__":
    main()
