# TFjs ACGAN Browser Training
Train an Auxiliary Classifier GAN (ACGAN) in a browser tab with TensorFlow.js:

[josephrocca.github.io/tfjs-acgan-browser-training/train.html](https://josephrocca.github.io/tfjs-acgan-browser-training/train.html)

In a "vanilla" GAN we give the generator some random numbers as input, and it converts them into an image. The discriminator is then given these images, along with *real* images, and outputs a "realness" score. Each training step, we alter the weights of the discriminator (with backpropagation) so that its realness score is more accurate, and we alter the weights of the generator so it's more likely to trick the discriminator. Once we've finished training the whole network, we can throw away the discriminator part and just feed random numbers into the generator, and it'll generate random images for us.

The difference with an ACGAN is that we also give the generator an input that corresponds to the class of the image that we want it to generate (in this case, a MNIST digit), and the discriminator has a second output (alongside the realness score) which tries to classify the images that we give it. Both the realness score and the classification are factored into the "loss" which we use to direct our weight updates. So once we've trained the network, we can feed random numbers to the generator, along with a specific class number, and it'll generate that specific digit class from our dataset, rather than just a random MNIST digit.

"Auxiliary" means "additional" or "supporting", so the name "Auxiliary Classifier GAN" refers to the fact that the discriminator has this additional classifier built in.

* [Here's a bare-bones demo](https://josephrocca.github.io/tfjs-acgan-browser-training/generate.html) of using the trained generator model for generation.
* Haven't generalised `.buildGenerator()` and `.buildDiscriminator()` beyond 28x28 pixel greyscale images
* Based on this repository (mostly a copy-paste, factoring out the node.js stuff): https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
* ACGAN paper: https://arxiv.org/abs/1610.09585
* I factored out the auxiliary classifier to create a second repo for a vanilla GAN trainer, but it suffered from [mode collapse](https://developers.google.com/machine-learning/gan/problems#mode-collapse) (which is mostly expected for vanilla Gans), and I haven't gotten around to implementing one of the common solutions yet.
