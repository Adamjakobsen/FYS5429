import numpy as np
import scipy.io
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


class PINN1D(nn.Module):
    """1D PINN model"""
    def __init__(self, num_layers, num_neurons, device):
        """Initialize the model"""
        super(PINN1D, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.device = device

        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers += [nn.Linear(num_neurons, 2)]

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def predict(self, x):
        """Predict the solution at x"""
        x = torch.tensor(x, dtype=torch.float64)
        return self.model(x).to('cpu').detach().numpy()


def model_loss(pinn, x, y_true, a, k, mu1, mu2, eps, b, h, D, x_left, x_right, T_ic):
    """
    Hybrid loss function

    """
    # PDE loss
    x_space, t_space = x[:, 0:1].clone().requires_grad_(
        True), x[:, 1:2].clone().requires_grad_(True)
    y_pred = pinn(torch.cat([x_space, t_space], dim=1))

    V, W = y_pred[:, 0:1], y_pred[:, 1:2]
    V_true, W_true = y_true[:, 0:1], y_true[:, 1:2]

    dVdt = torch.autograd.grad(V, t_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]
    dWdt = torch.autograd.grad(W, t_space, grad_outputs=torch.ones_like(
        W), create_graph=True, allow_unused=True)[0]

    dVdx = torch.autograd.grad(V, x_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]

    if dVdx is not None:
        dVdxx = torch.autograd.grad(dVdx, x_space, grad_outputs=torch.ones_like(
            dVdx), create_graph=True, allow_unused=True)[0]

    else:
        raise ValueError("Gradients dVdx are not computed correctly.")

    eq_V = dVdt - D * dVdxx + k * V * (V - a) * (V - 1) + W * V
    eq_W = dWdt - (eps + (mu1 * W) / (mu2 + V)) * (-W - k * V * (V - b - 1))


    loss_V = torch.mean(torch.square(eq_V))
    loss_W = torch.mean(torch.square(eq_W))

    #Data loss
    loss_data = torch.mean(torch.square(V - V_true)) + \
        torch.mean(torch.square(W - W_true))
    #Boundary loss
    loss_BC = boundary_loss(x,y_pred,x_left, x_right,pinn)
    #Initial condition loss
    loss_IC = initial_condition_loss(x,y_pred,V_true,W_true,T_ic,pinn)

    

    return loss_V + loss_W + loss_data + loss_BC + loss_IC

def boundary_loss(x, y,x_left, x_right, pinn):
    """Neumann boundary loss"""
    v_x = torch.autograd.grad(y[:, 0], x, create_graph=True, grad_outputs=torch.ones_like(y[:, 0]))[0][:, 0:1]
    on_boundary = (x[:, 0] == x_left) | (x[:, 0] == x_right)
    v_x_boundary = v_x[on_boundary]
    loss = torch.mean(v_x_boundary**2)
    return loss

def initial_condition_loss(x, y, observe_train, v_train,T_ic,pinn):
    """Initial condition loss"""
    T = x[:, -1].reshape(-1, 1)
    T_ic = torch.tensor(T_ic, dtype=torch.float64)
    idx_init = torch.where(torch.isclose(T, T_ic))[0]
    y_init = y[idx_init]
    v_init = v_train[idx_init]
    loss = torch.mean((y_init - v_init)**2)
    return loss


def train(model, optimizer, data_loader_train,data_loader_test, a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic):
    #Training
    model.train()
    for x, y_true in data_loader_train:
        x = x.to(device)
        y_true = y_true.to(device)
        x.requires_grad = True  # Set requires_grad to True

        optimizer.zero_grad()
        loss_train = model_loss(model, x, y_true, a, k, mu1, mu2,
                          eps, b, h, D, x_left, x_right, T_ic)
        loss_train.backward()
        optimizer.step()

        # Ensure all tensors are on the same device
        for param in model.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
    
    #Validation      
    model.eval()
    for x, y_true in data_loader_test:
        x = x.to(device)
        y_true = y_true.to(device)
        x.requires_grad = True
        loss_test = model_loss(model, x, y_true, a, k, mu1, mu2,eps, b, h, D, x_left, x_right, T_ic)

    

    return loss_train.item(),loss_test.item()


def generate_data():
    mat = scipy.io.loadmat('./data/data1D.mat')
    V = mat['Vsav']
    W = mat['Wsav']
    t = mat['t']
    x = mat['x']

    X, T = np.meshgrid(x, t)
    X = X.reshape(-1, 1)
    
    T = T.reshape(-1, 1)

    V = V.reshape(-1, 1)
    W = W.reshape(-1, 1)

    


    x_left = np.min(x)
    x_right = np.max(x)

    T_ic = np.min(t)

    return np.hstack((X, T)), np.hstack((V, W)), x_left, x_right, T_ic

def run_training(model, opt, data_loader_train,data_loader_test, a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic, n_epochs,learning_rate,w):
    pinn=model
    loss_test_history = []
    loss_train_history = []
    if opt == 'AdamW':
        optimizer = optim.AdamW(pinn.parameters(), lr=learning_rate,weight_decay=w)
    if opt == 'Adam':
        optimizer = optim.Adam(pinn.parameters(), lr=learning_rate, weight_decay=w)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    for epoch in range(n_epochs):
        loss_train,loss_test = train(pinn, optimizer, data_loader_train,data_loader_test,
                    a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic)
        scheduler.step()
        loss_train_history.append(loss_train)
        loss_test_history.append(loss_test)
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Train loss: {loss_train} ,test loss: {loss_test}")

    # Save the model
    torch.save(pinn.state_dict(), f"./pinn_aliev_panfilov_{n_epochs}_schedule_{opt}_{learning_rate}_{w}.pt")
    #Save the loss history
    print("Saving loss history to",f"loss_train_history_1D_{n_epochs}_schedule_{opt}_{learning_rate}_{w}.npy")
    np.save(f"loss_train_history_1D_{n_epochs}_schedule_{opt}_{learning_rate}_{w}.npy", loss_train_history)
    np.save(f"loss_test_history_1D_{n_epochs}_schedule_{opt}_{learning_rate}_{w}.npy", loss_test_history)
    return loss_train_history,loss_test_history


def main():
    #Set default float64
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    num_neurons = 32
    num_layers = 3
    n_epochs = 1#15000
    batch_size = 512

    

    # Generate data for training and testing
    X, Y, x_left, x_right, T_ic = generate_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=True, test_size=0.9)
    print("Size Y_train: ", Y_train.shape)
    print("Size Y_test: ", Y_test.shape)
    print("Max x is: ", np.max(X_train[:,0]))
    print("Min t is: ", np.min(X_train[:,1]))
    #from sklearn.preprocessing import StandardScaler
    #scaler=StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    
    #Add noise to training data
    #Y_train=Y_train + 0.01*np.random.randn(*Y_train.shape)

    dataset_train = torch.utils.data.TensorDataset(torch.tensor(
        X_train, dtype=torch.float64).to(device), torch.tensor(Y_train, dtype=torch.float64).to(device))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_test = torch.utils.data.TensorDataset(torch.tensor(
        X_test, dtype=torch.float64).to(device), torch.tensor(Y_test, dtype=torch.float64).to(device))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # Parameters for the Aliev-Panfilov model
    a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    eps = 0.002
    b = 0.15
    h = 0.1
    D = 0.1
    # Create PINN object
    pinn = PINN1D(num_layers, num_neurons, device).to(device)
    learning_rate = 0.005
    w = 0.01
    opt ="AdamW"
    print("Optimizer: ", opt)
    print("Learning rate: ", learning_rate)
    print("Weight decay: ", w)
    
    
    loss_train_history,loss_test_history= run_training(pinn, opt, data_loader_train,data_loader_test, a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic, n_epochs,learning_rate,w)
    # Test the model
    pinn.eval()
    
    Y_pred_test= pinn.predict(torch.tensor(X_test, dtype=torch.float64).to(device))
    
    #RMSE test
    RMSE_test=np.sqrt(np.mean((Y_pred_test-Y_test)**2))
    
    print(f"RMSE test: {RMSE_test}")


    Y_pred = pinn.predict(torch.tensor(X, dtype=torch.float64).to(device))

    mat = scipy.io.loadmat('./data/data_1d_left.mat')
    V = mat['Vsav']
    x = mat['x']
    # print("Mean squared error:", np.mean((Y_pred - Y_test) ** 2))
    V_pred = Y_pred[:, 0]
    V_pred = np.reshape(V_pred, (V.shape[0], V.shape[1]))

    print(f"Shape V_pred:{V_pred.shape}")
    print(f'Shape x: {x.shape}')
    print(f'shape V_GT: {V.shape}')
    plt.plot(x[0, :], V_pred[15, :], label="Predicted")
    plt.plot(x[0, :], V[15, :], label="True")
    plt.xlabel("x")
    plt.ylabel("V")
    plt.legend()
    plt.show()
    #Save y_pred
    np.save("y_pred_aliev_panfilov_20000_schedule_AdamW_0.005_0.01.npy", Y_pred)

    #plot loss
    plt.plot(np.linspace(0,n_epochs,len(loss_train_history)),loss_train_history,label="Train loss")
    plt.plot(np.linspace(0,n_epochs,len(loss_test_history)),loss_test_history,label="Test loss")
    plt.show()


if __name__ == "__main__":
    main()
