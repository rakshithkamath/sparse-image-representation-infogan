# Info GAN
In this repo we provide an implementation of InfoGAN for digital art creation using the CryptoPunk dataset

# Results
Our GAN is able to output images shown below (right) next to images from the original dataset (left)!
[image](static/images/crypto_vs_ours.jpg)


# Installation
```pip install -r requirements.txt```

# Usage
To run, simply call ```python src/train.py```.
Available cutumizations are also possible below:
```
usage: train.py [-h] [--epochs EPOCHS] [--print_every PRINT_EVERY] [--learning_rate_disc LEARNING_RATE_DISC] [--learning_rate_gen LEARNING_RATE_GEN] [--learning_rate_q LEARNING_RATE_Q]
                [--latent_dim LATENT_DIM] [--model_dir MODEL_DIR] [--dataset_name DATASET_NAME] [--batch_size BATCH_SIZE] [--num_categories NUM_CATEGORIES]
                [--num_continuous NUM_CONTINUOUS] [--disc_train_interval DISC_TRAIN_INTERVAL] [--noise_scale NOISE_SCALE] [--model_in_folder MODEL_IN_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to train.
  --print_every PRINT_EVERY
                        Number of epochs to train.
  --learning_rate_disc LEARNING_RATE_DISC
                        Learning rate for the discriminator
  --learning_rate_gen LEARNING_RATE_GEN
                        Learning rate for the generator
  --learning_rate_q LEARNING_RATE_Q
                        Learning rate for the generator
  --latent_dim LATENT_DIM
                        The number of elements of pure noise
  --model_dir MODEL_DIR
  --dataset_name DATASET_NAME
  --batch_size BATCH_SIZE
  --num_categories NUM_CATEGORIES
  --num_continuous NUM_CONTINUOUS
  --disc_train_interval DISC_TRAIN_INTERVAL
  --noise_scale NOISE_SCALE
  --model_in_folder MODEL_IN_FOLDER
                        Path to a folder containing disc / generator weights. The latest one will be loaded.
```

# References
- Used [this guide](https://machinelearningmastery.com/how-to-develop-an-information-maximizing-generative-adversarial-network-infogan-in-keras/) as a reference when developing InfoGAN
- CryptoPunk training data taken from [this resource](https://github.com/larvalabs/cryptopunks/blob/master/punks.png)