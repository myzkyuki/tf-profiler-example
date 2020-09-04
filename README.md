# TensorFlow Profiler example
Profile MNIST train code with TensorFlow Profiler.

## Install
```bash
$ pip3 install tensorflow==2.3.0
$ pip3 install tensorboard==2.3.0
$ pip3 install tensorboard-plugin-profile==2.3.0
```

## Train
Run following code to train MNIST.
```bash
$ python3 train.py
```

Then we can see profile logs with TensorBoard.
```bash
$ tensorboard --logdir ./logs
```