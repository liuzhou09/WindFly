# RA-Net

RA-Net defines a gray-box UAV acceleration model whose MLP predicts a three-axis residual from a 10-step feature window. [model.py](model.py) contains the network definition, and [evaluation figures](outputs/evaluation/) show selected acceleration predictions.

Install PyTorch and import `GrayBoxDynamicsModel` from `model.py` to inspect the model interface. The included model definition has no pretrained parameters; training and evaluation programs will be added in a future update.
