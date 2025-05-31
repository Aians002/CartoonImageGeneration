Cartoon Generator Model

This project implements a cartoon image generator that creates 128x128 cartoon faces from 18 facial attributes, using a hybrid approach combining a Conditional Variational Autoencoder (CVAE) and a Conditional Generative Adversarial Network (cGAN). The model outputs both colored images and corresponding outlines, trained on a preprocessed synthetic dataset.

Dataset

The model uses the CartoonSet100k dataset, a synthetic cartoon face dataset provided by Google LLC under the Creative Commons Attribution 4.0 International License. The dataset includes images and associated attribute files, which are preprocessed into tensors for training.

Preprocessing

Images are resized to 128x128, normalized, and paired with outlines generated via Canny edge detection. Attributes from CSV files are converted into tensors, resulting in a structured dataset stored in cartoonset100k_tensors.

Model Architecture

Generator: Encodes attributes into a latent space, then uses separate structure and texture branches with residual blocks to produce outlines and colored images.
Discriminator: A convolutional network distinguishing real from generated images.
Attribute Predictor: Ensures generated images align with input attributes.
Training

Trained for 200 epochs with a batch size of 32.
Uses Adam optimizers (lr=5e-4 for generator, 2e-4 for discriminator) with learning rate scheduling.
Combines multiple losses: color, outline, perceptual, adversarial, and attribute prediction.
Usage

Preprocess the dataset: python preProcess.py
Train the model: python train.py
Adjust hyperparameters in the configuration section of the scripts as needed.
