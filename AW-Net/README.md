# AW-Net

AW-Net studies a recurrent UAV flight policy that uses depth observations and disturbance-aware acceleration constraints. [model.py](model.py) defines the policy, and [main_awnet.py](main_awnet.py) provides the training objective and command-line options.

To inspect the available options, run from this directory:

```bash
python main_awnet.py --help
```

The CUDA simulator and runnable training setup will be added in a future update.
