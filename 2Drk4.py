import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.ndimage import convolve1d
from scipy import ndimage


"""
Python implementation of:
Data generation for the Aliev-Panfilov model from EP-PINNs




https://github.com/martavarela/EP-PINNs/blob/main/Matlab_code/AlievPanfilov1D_RK.m

"""


def AlPan(V, W, a, D,Ia):
    # Parameters

    # a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    eps = 0.002
    b = 0.15
    h = 0.1  # cell length [mm]
    

    del2V = ndimage.laplace(V, mode='nearest')/h**2


    dVdt = (-k*V*(V-a)*(V-1) - W*V) + 4*D*del2V + Ia
    dWdt = (eps + mu1*W/(mu2+V))*(-W-k*V*(V-b-1))

    return dVdt, dWdt


def RK4(V, W, dt, a, D,Ia):
    #print("shape Ia: ", Ia.shape)
    k1_V = AlPan(V, W, a, D,Ia)[0]
    k1_W = AlPan(V, W, a, D,Ia)[1]

    k2_V = AlPan(V + k1_V*dt/2, W + k1_W*dt/2, a, D,Ia)[0]
    k2_W = AlPan(V + k1_V*dt/2, W + k1_W*dt/2, a, D,Ia)[1]

    k3_V = AlPan(V + k2_V*dt/2, W + k2_W*dt/2, a, D,Ia)[0]
    k3_W = AlPan(V + k2_V*dt/2, W + k2_W*dt/2, a, D,Ia)[1]

    k4_V = AlPan(V + k3_V*dt, W + k3_W*dt, a, D,Ia)[0]
    k4_W = AlPan(V + k3_V*dt, W + k3_W*dt, a, D,Ia)[1]

    V = V + (k1_V + 2*k2_V + 2*k3_V + k4_V)*dt/6
    W = W + (k1_W + 2*k2_W + 2*k3_W + k4_W)*dt/6
    return V, W


def solve(n_cells, BCL, ncyc, dt, stim_dur, Va, a, D):

    # Allow boundary conditions to be set
    X = n_cells + 2  # Number of cells + 2 boundary cells
    Y = n_cells + 2  # Number of cells + 2 boundary cells

    stigmeo = np.zeros((X,Y), dtype=bool)
    
    stigmeo[0:5,:] = True  # Where the stimulus is applied
    

    gathert = round(1/dt)  # Number of time steps

    t_end = ncyc*BCL  # End time [AU]

    # Initial conditions V and W
    V = np.zeros((X,Y))
    #V[1:X,1:X] = 0

    W = np.zeros((X,Y)) + 0.01
    #W[1:X,1:X] = 0.01

    V_save = np.zeros((n_cells,n_cells, int(np.ceil(t_end/dt))))
    W_save = np.zeros((n_cells,n_cells, int(np.ceil(t_end/dt))))
    

    stim_count = 0
    it_count = 0
    y = np.vstack((V, W))
    print(f" t_end: {t_end}")
    print(f"X: {X}")

    for t in np.arange(dt, t_end+dt, dt):
        it_count += 1

        #print(f"V shape: {V.shape}")
        if t >= stim_count*BCL and stim_count < ncyc:
            Ia = 0.1*np.zeros_like(V)
            Ia[stigmeo] = Va

        if t >= BCL*stim_count+stim_dur:
            stim_count += 1
            Ia = np.zeros_like(V)

        V, W = RK4(V, W, dt, a, D,Ia)
        # print(f"y shape: {y.shape}")

        V[1,:] = V[0,:]
        V[:,1] = V[:,0]
        V[-2,:] = V[-1,:]
        V[:,-2] = V[:,-1]
        
        if it_count % gathert == 0:

            V_save[:,:, round(it_count/gathert)] = V[1:-1,1:-1].T
            W_save[:,:, round(it_count/gathert)] = W[1:-1,1:-1].T
        


    return V_save, W_save





def main():

    n_cells = 100  # Number of cells
    BCL = 100  # Basic Cycle Length in [AU]: time between repeated stimuli
    ncyc = 1  # Number of cycles: Number times the cell is stimulated

    dt = 0.005  # Time step [AU]

    stim_dur = 1  # Stimulus duration [AU]

    Va = 0.1  # Value of V when the stimulus is applied
    a=0.01
    D=0.05
    V_save, W_save = solve(n_cells, BCL, ncyc, dt, stim_dur, Va, a, D)
    V_save = V_save[:, :,:70]
    W_save = W_save[:, :,:70]



    print(f"V_save shape: {V_save.shape}")

    np.save(f"./data/2D_V_a{a}_D{D}.npy", V_save)
    np.save(f"./data/2D_W_a{a}_D{D}.npy", W_save)

    
    
    V = V_save
    W = W_save
    t = t = np.arange(1,71,1)
    x = np.arange(0.1,10.1,0.1)
    y = np.arange(0.1,10.1,0.1)
    #Save .mat keys: V, W, t, x, y
    from scipy.io import savemat
    savemat(f"./data/data2D.mat", {"V":V, "W":W, "t":t, "x":x, "y":y})
    



if __name__ == "__main__":
    main()
