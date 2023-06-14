# Project repo for FYS5429-Advanced Machine Learning for Physical Sciences

## Requirements

All requirements can b installed by creating an environment using:
```
conda create --name <env> --file heartpinn_env.txt
```

## Training

### 1D cable

`pinn1D.py` can be run as is, the output will be the RMSE for the test data:
```
python pinn1D.py
```

### 2D square domain
´pinn2D.py' takes command line arguments `train` and `predict`. The code uses Pytorch as backlend, which has to be specified. For training run:

```
DDE_BACKEND=pytorch python pinn2D.py train
```

### MRI-based geometry

Similar to the 2D square run:
```
DDE_BACKEND=pytorch python heartpinn.py train
```


