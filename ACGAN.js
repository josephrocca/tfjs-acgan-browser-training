// This code is based on: https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
// Which in turn was based on: https://github.com/keras-team/keras/blob/2.4.0/examples/mnist_acgan.py
// For background of ACGAN, see the paper: https://arxiv.org/abs/1610.09585

export default class ACGAN {

  constructor(opts={}) {

    this.xTrain = null;
    this.yTrain = null;

    // Number of classes in the MNIST dataset.
    this.numClasses = 10;

    // MNIST image size.
    this.imageSize = 28;

    this.batchSize = opts.batchSize ?? 100;
    this.latentSize = opts.latentSize ?? 100;

    // See section 3.4 here: https://arxiv.org/pdf/1606.03498.pdf
    this.softOne = 0.95;

    // Adam parameters suggested here: https://arxiv.org/abs/1511.06434
    this.learningRate = opts.learningRate ?? 0.0002,
    this.adamBeta1 = opts.adamBeta1 ?? 0.5,

    // Build the discriminator.
    this.discriminator = this.buildDiscriminator();
    this.discriminator.compile({
      optimizer: tf.train.adam(this.learningRate, this.adamBeta1),
      loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
    });
    this.discriminator.summary();

    // Build the generator.
    this.generator = this.buildGenerator(this.latentSize);
    this.generator.summary();

    this.optimizer = tf.train.adam(this.learningRate, this.adamBeta1);
    this.combinedModel = this.buildCombinedModel(this.latentSize, this.generator, this.discriminator, this.optimizer);
    
  }




  async setTrainingData(opts) {
    let images, labels;
    if(opts === "MNIST") {
      let data = await loadMnistData();
      images = data.images;
      labels = data.labels;
    } else {
      images = opts.images;
      labels = opts.labels;
    }
    this.xTrain = images;
    this.yTrain = tf.expandDims(labels.argMax(-1), -1);
  }




  /**
   * Build the generator part of ACGAN.
   *
   * The generator of ACGAN takes two inputs:
   *
   *   1. A random latent-space vector (the latent space is often referred to as "z-space" in GAN literature).
   *   2. A label for the desired image category (0, 1, ..., 9).
   *
   * It generates one output: the generated (i.e., fake) image.
   *
   * @param {number} latentSize Size of the latent space.
   * @returns {tf.LayersModel} The generator model.
   */
  buildGenerator(latentSize) {
    tf.util.assert(latentSize > 0 && Number.isInteger(latentSize), `Expected latent-space size to be a positive integer, but got ${latentSize}.`);

    const cnn = tf.sequential();

    // The number of units is chosen so that when the output is reshaped and fed through the subsequent conv2dTranspose layers, the tensor
    // that comes out at the end has the exact shape that matches MNIST images ([28, 28, 1]).
    cnn.add(tf.layers.dense({units: 3 * 3 * 384, inputShape: [latentSize], activation: 'relu'}));
    cnn.add(tf.layers.reshape({targetShape: [3, 3, 384]}));

    // Upsample from [3, 3, ...] to [7, 7, ...].
    cnn.add(tf.layers.conv2dTranspose({filters: 192, kernelSize: 5, strides: 1, padding: 'valid', activation: 'relu', kernelInitializer: 'glorotNormal'}));
    cnn.add(tf.layers.batchNormalization());

    // Upsample to [14, 14, ...].
    cnn.add(tf.layers.conv2dTranspose({filters: 96, kernelSize: 5, strides: 2, padding: 'same', activation: 'relu', kernelInitializer: 'glorotNormal'}));
    cnn.add(tf.layers.batchNormalization());

    // Upsample to [28, 28, ...].
    cnn.add(tf.layers.conv2dTranspose({filters: 1, kernelSize: 5, strides: 2, padding: 'same', activation: 'tanh', kernelInitializer: 'glorotNormal'}));

    // Unlike most TensorFlow.js models, the generator part of an ACGAN has
    // two inputs:
    //   1. The latent vector that is used as the "seed" of the fake image generation.
    //   2. A class label that controls which of the ten MNIST digit classes the generated fake image is meant to belong to.

    // This is the z space commonly referred to in GAN papers.
    const latent = tf.input({shape: [latentSize]});

    // The desired label of the generated image, an integer in the interval [0, this.numClasses).
    const imageClass = tf.input({shape: [1]});

    // The desired label is converted to a vector of length `latentSize` through embedding lookup.
    const classEmbedding = tf.layers.embedding({inputDim: this.numClasses, outputDim: latentSize, embeddingsInitializer: 'glorotNormal'}).apply(imageClass);

    // Hadamard product between z-space and a class conditional embedding.
    const h = tf.layers.multiply().apply([latent, classEmbedding]);

    const fakeImage = cnn.apply(h);
    return tf.model({inputs: [latent, imageClass], outputs: fakeImage});
  }










  /**
   * Build the discriminator part of ACGAN.
   *
   * The discriminator model of ACGAN takes the input: an image of MNIST format, of shape [batchSize, 28, 28, 1].
   *
   * It gives two outputs:
   *
   *   1. A sigmoid probability score between 0 and 1, for whether the discriminator judges the input image to be real (close to 1)
   *      or fake (closer to 0).
   *   2. Softmax probability scores for the 10 MNIST digit categories, which is the discriminator's 10-class classification result
   *      for the input image.
   *
   * @returns {tf.LayersModel} The discriminator model.
   */
  buildDiscriminator() {
    const cnn = tf.sequential();

    cnn.add(tf.layers.conv2d({filters: 32, kernelSize: 3, padding: 'same', strides: 2, inputShape: [this.imageSize, this.imageSize, 1]}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));

    cnn.add(tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same', strides: 1}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));

    cnn.add(tf.layers.conv2d({filters: 128, kernelSize: 3, padding: 'same', strides: 2}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));

    cnn.add(tf.layers.conv2d({filters: 256, kernelSize: 3, padding: 'same', strides: 1}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));

    cnn.add(tf.layers.flatten());

    const image = tf.input({shape: [this.imageSize, this.imageSize, 1]});
    const features = cnn.apply(image);

    // Unlike most TensorFlow.js models, the discriminator has two outputs.

    // The 1st output is the probability score assigned by the discriminator to how likely the input example is a real MNIST image (as versus
    // a "fake" one generated by the generator).
    const realnessScore = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features);
    // The 2nd output is the softmax probabilities assign by the discriminator for the 10 MNIST digit classes (0 through 9). "aux" stands for "auxiliary"
    // (the namesake of ACGAN) and refers to the fact that unlike a standard GAN (which performs just binary real/fake classification), the discriminator
    // part of ACGAN also performs multi-class classification.
    const aux = tf.layers.dense({units: this.numClasses, activation: 'softmax'}).apply(features);

    return tf.model({inputs: image, outputs: [realnessScore, aux]});
  }











  /**
   * Build a combined ACGAN model.
   *
   * @param {number} latentSize Size of the latent vector.
   * @param {tf.SymbolicTensor} imageClass Symbolic tensor for the desired image class. This is the other input to the generator.
   * @param {tf.LayersModel} generator The generator.
   * @param {tf.LayersModel} discriminator The discriminator.
   * @param {tf.Optimizer} optimizer The optimizer to be used for training the combined model.
   * @returns {tf.LayersModel} The combined ACGAN model, compiled.
   */
  buildCombinedModel(latentSize, generator, discriminator, optimizer) {
    // Latent vector. This is one of the two inputs to the generator.
    const latent = tf.input({shape: [latentSize]});
    // Desired image class. This is the second input to the generator.
    const imageClass = tf.input({shape: [1]});
    // Get the symbolic tensor for fake images generated by the generator.
    let fakeImage = generator.apply([latent, imageClass]);
    let aux;

    // We only want to be able to train generation for the combined model.
    discriminator.trainable = false;
    [fakeImage, aux] = discriminator.apply(fakeImage);
    const combined = tf.model({inputs: [latent, imageClass], outputs: [fakeImage, aux]});
    combined.compile({optimizer, loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']});
    combined.summary();
    return combined;
  }













  /**
   * Train the discriminator for one step.
   *
   * In this step, only the weights of the discriminator are updated. The generator is not involved.
   *
   * The following steps are involved:
   *
   *   - Slice the training features and to get batch of real data.
   *   - Generate a random latent-space vector and a random label vector.
   *   - Feed the random latent-space vector and label vector to the generator and let it generate a batch of generated (i.e., fake) images.
   *   - Concatenate the real data and fake data; train the discriminator on the concatenated data for one step.
   *   - Obtain and return the loss values.
   *
   * @param {tf.Tensor} xTrain A tensor that contains the features of all the training examples.
   * @param {tf.Tensor} yTrain A tensor that contains the labels of all the training examples.
   * @param {number} batchStart Starting index of the batch.
   * @param {number} batchSize Size of the batch to draw from `xTrain` and `yTrain`.
   * @param {number} latentSize Size of the latent space (z-space).
   * @param {tf.LayersModel} generator The generator of the ACGAN.
   * @param {tf.LayersModel} discriminator The discriminator of the ACGAN.
   * @returns {number[]} The loss values from the one-step training as numbers.
   */
  async trainDiscriminatorOneStep(xTrain, yTrain, batchStart, batchSize, latentSize, generator, discriminator) {

    const imageBatch = xTrain.slice(batchStart, batchSize);
    const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');

    // Latent vectors.
    let zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    let sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');

    const generatedImages = generator.predict([zVectors, sampledLabels], {batchSize: batchSize});

    const x = tf.concat([imageBatch, generatedImages], 0);

    const y = tf.tidy(() => {
      return tf.concat([tf.ones([batchSize, 1]).mul(this.softOne), tf.zeros([batchSize, 1])]);
    });

    const auxY = tf.concat([labelBatch, sampledLabels], 0);

    const losses = await discriminator.trainOnBatch(x, [y, auxY]);
    tf.dispose([x, y, auxY]);
    return losses;
  }













  /**
   * Train the combined ACGAN for one step.
   *
   * In this step, only the weights of the generator are updated.
   *
   * @param {number} batchSize Size of the fake-image batch to generate.
   * @param {number} latentSize Size of the latent space (z-space).
   * @param {tf.LayersModel} combined The instance of tf.LayersModel that combines the generator and the discriminator.
   * @returns {number[]} The loss values from the combined model as numbers.
   */
  async trainCombinedModelOneStep(batchSize, latentSize, combinedModel) {
    
    // Make new latent vectors.
    const zVectors = tf.randomUniform([batchSize, latentSize], -1, 1); // <-- noise
    const sampledLabels = tf.randomUniform([batchSize, 1], 0, this.numClasses, 'int32').asType('float32');

    // We want to train the generator to trick the discriminator.
    // For the generator, we want all the {fake, not-fake} labels to say not-fake.
    const trick = tf.ones([batchSize, 1]).mul(this.softOne);

    const losses = await combinedModel.trainOnBatch([zVectors, sampledLabels], [trick, sampledLabels]);
    tf.dispose([zVectors, sampledLabels, trick]);
    return losses;
  }












  async train() {

    const numBatches = Math.ceil(this.xTrain.shape[0] / this.batchSize);

    for(let batch = 0; batch < numBatches; batch++) {

      const actualBatchSize = (batch + 1) * this.batchSize >= this.xTrain.shape[0] ? (this.xTrain.shape[0] - batch * this.batchSize) : this.batchSize;
      const dLoss = await this.trainDiscriminatorOneStep(this.xTrain, this.yTrain, batch * this.batchSize, actualBatchSize, this.latentSize, this.generator, this.discriminator);

      // Here we use 2 * actualBatchSize here, so that we have the generator optimizer over an identical number of images as the discriminator.
      const gLoss = await this.trainCombinedModelOneStep(2 * actualBatchSize, this.latentSize, this.combinedModel);

      console.log(`batch ${batch + 1}/${numBatches}: dLoss = ${dLoss[0].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}`);
    }

  }







  async downloadGeneratorModel() {
    await this.generator.save("downloads://model");
  }







  // Generate a set of examples (one for each class) using the generator model of the ACGAN.
  // Outputs result to given canvas, or returns image data as a Uint8ClampedArray.
  async generate(canvas) {
    tf.util.assert(this.generator.inputs.length === 2, `Expected model to have exactly 2 symbolic inputs, but there are ${this.generator.inputs.length}`);

    const combinedFakes = tf.tidy(() => {
      
      const latentVectorLength = this.generator.inputs[0].shape[1];
      const numRepeats = 10; // <-- number of times to tile the single latent vector for. Used for generating a batch of fake MNIST images.
      
      const latentDims = latentVectorLength;
      const zs = new Array(latentDims).fill(0).map(_ => Math.random());
      const singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
      const latentVectors = singleLatentVector.tile([numRepeats, 1]); // <-- the tiled latent-space vector, of shape [numRepeats, latentDim].

      // Generate one fake image for each digit.
      const sampledLabels = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
      // The output has pixel values in the [-1, 1] interval. Normalize it to the unit interval ([0, 1]).
      const t0 = tf.util.now();
      const generatedImages = this.generator.predict([latentVectors, sampledLabels]).add(1).div(2);
      generatedImages.dataSync();  // For accurate timing benchmark.
      const elapsed = tf.util.now() - t0;
      console.log(`Generation took ${elapsed.toFixed(2)} ms`);
      
      // Concatenate the images horizontally into a single image.
      return tf.concat(tf.unstack(generatedImages), 1);
    });

    let uint8Clamped = await tf.browser.toPixels(combinedFakes, canvas);
    tf.dispose(combinedFakes);

    return uint8Clamped;
  }




}





async function loadMnistData() {
  // This code is based on code from here: https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
  const IMAGE_H = 28;
  const IMAGE_W = 28;
  const IMAGE_SIZE = IMAGE_H * IMAGE_W;
  const NUM_CLASSES = 10;
  const NUM_DATASET_ELEMENTS = 65000;

  const NUM_TRAIN_ELEMENTS = 55000;
  const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

  const MNIST_IMAGES_SPRITE_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
  const MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

  let datasetImages;
  
  // Make a request for the MNIST sprited image.
  const img = new Image();
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const imgRequest = new Promise((resolve, reject) => {
    img.crossOrigin = "";
    img.onerror = reject;
    img.onload = () => {
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;

      const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

      const chunkSize = 5000;
      canvas.width = img.width;
      canvas.height = chunkSize;

      for(let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
        const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize);
        ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for(let j = 0; j < imageData.data.length / 4; j++) {
          // All channels hold an equal value since the image is grayscale, so
          // just read the red channel.
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
      }
      datasetImages = new Float32Array(datasetBytesBuffer);

      resolve();
    };
    img.src = MNIST_IMAGES_SPRITE_PATH;
  });

  const labelsRequest = fetch(MNIST_LABELS_PATH);
  const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

  let datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

  // Slice the the images and labels into train and test sets.
  let trainImages = datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
  let trainLabels = datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);

  // image data is in range [0, 1] and so we convert to [-1, 1] with .sub(0.5).mul(2)
  const images = tf.tensor4d(trainImages, [trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]).sub(0.5).mul(2);
  const labels = tf.tensor2d(trainLabels, [trainLabels.length / NUM_CLASSES, NUM_CLASSES]);

  // images: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
  // labels: The one-hot encoded labels tensor, of shape `[numTrainExamples, 10]`.
  return {images, labels};
}