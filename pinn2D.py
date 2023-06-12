import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
import torch as torch
import sys as sys
import scipy.io


class PINN():
    def __init__(self):
        """
        Initialize the class

        The parameters k, a, b, and eps determine the shape of the action potential and the refractory period of the cell. 
        The parameter mu1 determines the strength of the interaction between V and W, 
        while mu2 determines the level of V at which W has half of its maximal effect on the dynamics of the system.

        """
        self.a = 0.01
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1
        self.D = 0.1

    def pde2d(self, x, y):
        """
        Function that defines the PDE

        args:
            x: input data (x,t)
            y: output data (V,W)

        returns:
            Residual of the PDE

        """
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        # Coupled PDE+ODE Equations
        eq_a = dv_dt - self.D*(dv_dxx + dv_dyy) + \
            self.k*V*(V-self.a)*(V-1) + W*V
        eq_b = dw_dt - (self.eps + (self.mu1*W)/(self.mu2+V)) * \
            (-W - self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def get_data(self, filename):
        mat = scipy.io.loadmat(filename)
        t, x, y, V, W = mat['t'], mat['x'], mat['y'], mat['V'], mat['W']
        self.min_x, self.max_x = np.min(x), np.max(x)
        self.min_y, self.max_y = np.min(y), np.max(y)
        self.min_t, self.max_t = np.min(t), np.max(t)
        X, T, Y = np.meshgrid(x, t, y)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = V.reshape(-1, 1)
        W = W.reshape(-1, 1)
        return np.hstack((X, Y, T)), V, W

    def BC(self, geomtime):
        """
        Boundary condition function.(No flux Neumann BC)

        The NeumannBC class takes four arguments: 
        the geometry and time (combined into a single geomtime object), 
        a function that specifies the boundary condition value (in this case, a function that returns an array of zeros with the same length as the input x), 
        a function that specifies whether a point is on the boundary (in this case, a lambda function that returns True for all points on the boundary), 
        and an optional component argument that specifies the component of the solution to which the boundary condition applies (in this case, 0, which means the first component of the solution).

        args:
            geomtime : geometry of the problem

        returns:
            Boundary condition function for the PDE
        """
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros(
            (len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        return bc

    def IC(self, observe_train, v_train):
        """
        Initial condition function.




        args:
            observe_train: training data (x,t)
            v_train: training data (V,W)

        returns:
            Initial condition function for the PDE

        """

        T_ic = observe_train[:, -1].reshape(-1, 1)
        idx_init = np.where(np.isclose(T_ic, 1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init, v_init, component=0)

    def geotime(self):
        """
        Function that defines the geometry and time domain

        returns:
            Geometry and time domain of the problem
        """
        geom = dde.geometry.Rectangle([self.min_x, self.min_y], [
                                      self.max_x, self.max_y])
        timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime


def main():
    pinn = PINN()
    #Get data and train test split
    X, v, w = pinn.get_data('./data/data2D.mat')
    X_train, X_test, v_train, v_test, w_train, w_test = train_test_split(
        X, v, w, test_size=0.9)
    data_list = [X, X_train, v_train, v]

    # Generate geometry and time domain
    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    # Define initial condition for the observation points
    ic = pinn.IC(X_train, v_train)
    # Define boundary condition for the geometry and time domain
    bc = pinn.BC(geomtime)
    # Create TimePDE object
    input_data = [bc, ic, observe_v]
    data = dde.data.TimePDE(geomtime,
                            pinn.pde2d,
                            input_data,
                            num_domain=40000,
                            num_boundary=4000,
                            num_test=10000,
                            anchors=X_train)
    # Define the neural network architecture
    n_neurons = 60
    n_layers = 5
    n_epochs = 6000

    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], "tanh", "Glorot normal")
    #net.regularizer = ("l2", 0.01)

# Set model_save_path to save the model
    import os
    save_path = os.getcwd()+f"/models/AP2D2_model_{n_neurons}x{n_layers}_full"

    if sys.argv[1] == "train":
        init_weights = [0, 0, 0, 0, 1]
        checker = dde.callbacks.ModelCheckpoint(
            save_path, save_better_only=True, period=2000)

        model = dde.Model(data, net)
        #resampler = dde.callbacks.PDEPointResampler(period=1000)
        # Phase1
        model.compile("adam", lr=0.0005, loss_weights=init_weights)
        losshistory, train_state = model.train(
            epochs=15000, model_save_path=save_path)
        # Phase 2

        model.compile("adam", lr=0.0005)
        losshistory, train_state = model.train(
            epochs=150000, model_save_path=save_path)
        # Phase 3
        model.compile("L-BFGS-B")
        losshistory, train_state = model.train(model_save_path=save_path)

        #losshistory, train_state = model.train(
        #    epochs=n_epochs, model_save_path=save_path, callbacks=[checker])
        # model.compile("L-BFGS-B")

        # losshistory, train_state = model.train(model_save_path=save_path,callbacks=[checker])

        # Save and plot the loss history
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # Save the model
        model.save(save_path)

    if sys.argv[1] == "predict":
        torch.device("cpu")
        dde.config.default_device = "cpu"

        model = dde.Model(data, net)
        model.compile("L-BFGS-B")
        # model.compile("adam", lr=0.0001)
        model.restore(save_path+"-"+input("Enter model checkpoint: ")+".pt")
        #RMSE on test data
        pred = model.predict(X_test)
        RMSE = np.sqrt(np.mean((pred[:, 0:1]-v_test)**2))
        print("RMSE on test data: ", RMSE)
        from plot2d import generate_2D_animation, plot_2D_grid, generate_2D_animation_subplots
        plot_2D_grid(data_list, pinn, model, "planar_wave")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dde.config.default_device=device
    dde.config.set_random_seed(42)
    dde.config.set_default_float("float32")
    # torch.cuda.set_per_process_memory_fraction(0.9)

    main()
