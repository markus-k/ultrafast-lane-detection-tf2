# Ultra-Fast Lane Detection in Tensorflow 2 / Lite

This is a TensorFlow 2 and TensorFlow Lite implementation of the [Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757). The aim of this implementation is to run the network on a Google Edge TPU in an embedded device.

The model code in this repository is taken from the [original code](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and adapted to run with TensorFlow instead of pytorch. The ResNet18 implementation used is [calmisential's TF2.0 ResNet](https://github.com/calmisential/TensorFlow2.0_ResNet).
