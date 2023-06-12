import gc
import deepxde as dde
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch as torch
import sys as sys
import scipy.io
import os



class CustomPointCloud(dde.geometry.PointCloud):
    """Custom point cloud class for the heart geometry"""
    def __init__(self, points, boundary_points, boundary_normals):
        super(CustomPointCloud, self).__init__(
            points, boundary_points, boundary_normals)

    def compute_k_nearest_neighbors(self, x, k=3):
        # Compute the k-nearest neighbors for each boundary point
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='auto').fit(self.boundary_points)
        distances, indices = nbrs.kneighbors(x)
        return indices

    def boundary_normal(self, x):
        k = 3  # number of neighbors
        indices = self.compute_k_nearest_neighbors(x, k)

        normals = np.zeros_like(x)
        for i, idx in enumerate(indices):

            normal = self.boundary_normals[idx[0]]

            normals[i] = normal

        return normals


class PINN():
    def __init__(self):
        """
        Initialize the class

        The parameters k, a, b, and eps determine the shape of the action potential and the refractory period of the cell. 
        The parameter mu1 determines the strength of the interaction between V and W, 
        while mu2 determines the level of V at which W has half of its maximal effect on the dynamics of the system.

        """
        self.a = 0.15
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1
        self.D = 1

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

    def get_data(self):
        from utils import get_data, get_boundary
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        vertices, triangles, vm = get_data()

        self.vertices = vertices
        self.triangles = triangles
        self.vm_full = vm
        print("self.triangles shape:", self.triangles.shape)
        self.vm = vm[:60, :]
        x = vertices[:, 0]
        y = vertices[:, 1]
        t = np.linspace(0, 600, 121)[:60]

        X, T = np.meshgrid(x, t)
        Y, T = np.meshgrid(y, t)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        V = self.vm.reshape(-1, 1)
        vertices_boundary, triangles_boundary = get_boundary()

        self.vertices_boundary = vertices_boundary
        self.triangles_boundary = triangles_boundary

        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)

        return np.hstack((X, Y, T)), np.hstack((X_boundary, Y_boundary, T_boundary)), V

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

        idx_init = np.where(np.isclose(T_ic, 5, rtol=1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]

        return dde.PointSetBC(observe_init, v_init, component=0)

    def geotime(self):
        """
        Function that defines the geometry and time domain

        returns:
            Geometry and time domain of the problem
        """

        self.boundary_normals = np.load("normals.npy")
        # remove points from vertices that are on the boundary
        vertices_expanded = self.vertices[:, np.newaxis]
        boundary_vertices_expanded = self.vertices_boundary[np.newaxis, :]

        is_vertex_on_boundary = np.any(
            np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        self.unique_vertices = self.vertices[~is_vertex_on_boundary]

        geom = CustomPointCloud(
            self.unique_vertices, self.vertices_boundary, self.boundary_normals)
        timedomain = dde.geometry.TimeDomain(0, 600)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime


def main():
    # Create PINN object
    pinn = PINN()

    # Generate data for training and testing and scale input features
    X, X_boundary, v = pinn.get_data()
    X_train, X_test, v_train, v_test = train_test_split(
        X, v, test_size=0.8)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    data_list = [X, X_train, v_train, v]

    # Generate geometry and time domain
    geomtime = pinn.geotime()
    #Define IC and BC
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, ic, observe_v]
    # Create TimePDE object
    data = dde.data.TimePDE(geomtime,
                            pinn.pde2d,
                            input_data,
                            num_domain=20000,
                            num_boundary=1000)
    # Define the neural network architecture
    n_neurons = 50
    n_layers = 4
    n_epochs = 6000
    activations = ["tanh", "tanh", "tanh",
                   "tanh", "tanh", "tanh"]
    net = dde.maps.FNN([3] + [100] + n_layers * [n_neurons] +
                       [2], activations, "Glorot normal")
    net.regularizer = ("l2", 0.1)



# Set model_save_path to save the model
    import os
    save_path = os.getcwd()+f"/models/heart_model_{n_neurons}x{n_layers}"

    if sys.argv[1] == "train":
        init_weights = [0, 0, 0, 1, 0]
        checker = dde.callbacks.ModelCheckpoint(
            save_path, save_better_only=True, period=2000)

        model = dde.Model(data, net)


        # Phase1
        model.compile("adamw", lr=0.0005, loss_weights=init_weights)
        losshistory, train_state = model.train(
            iterations=10000, model_save_path=save_path)

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        
        # Phase 2
        weights_phase2 = [0, 0, 0, 1, 1]
        model.compile("adamw", lr=0.0005, loss_weights=weights_phase2)
        losshistory, train_state = model.train(
            iterations=10000, model_save_path=save_path)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        # Phase 3
        weights_phase3 = [1, 1, 1, 1, 1]
        model.compile("adamw", lr=0.0005, loss_weights=weights_phase3)
        losshistory, train_state = model.train(
            iterations=20000, model_save_path=save_path)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        model.save(save_path)

    if sys.argv[1] == "predict":
        torch.device("cpu")
        dde.config.default_device = "cpu"

        model = dde.Model(data, net)
        model.compile("adam", lr=0.0005)
        # model.compile("adam", lr=0.0001)
        model.restore(save_path+"-"+input("Enter model checkpoint: ")+".pt")
        from plot import generate_2D_animation, plot_2D, animate_absolute_error
        # plot_2D(pinn, model)
        #generate_2D_animation(pinn, model)
        # plot_2D_grid(data_list, pinn, model, "planar_wave")
        #RMSE
       
        pred_test = model.predict(X_test) 
        RMSE = np.sqrt(np.mean((pred_test-v_test)**2))
        print("RMSE: ", RMSE)


if __name__ == "__main__":
    dde.config.set_random_seed(42)
    torch.cuda.set_per_process_memory_fraction(1.0)

    dde.config.set_default_float("float32")

    main()