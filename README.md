# TFjs ACGAN Browser Training
Train an Auxiliary Classifier GAN (ACGAN) in a browser tab with TensorFlow.js

* Train: [josephrocca.github.io/tfjs-acgan-browser-training/train.html](https://josephrocca.github.io/tfjs-acgan-browser-training/train.html)
* Generate: [josephrocca.github.io/tfjs-acgan-browser-training/generate.html](https://josephrocca.github.io/tfjs-acgan-browser-training/generate.html)
* Haven't generalised `.buildGenerator()` and `.buildDiscriminator()` beyond 28x28 pixel greyscale images
* Based on this repository (mostly a copy-paste, factoring out the node.js stuff): https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
* ACGAN paper: https://arxiv.org/abs/1610.09585
