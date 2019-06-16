# Neural-net experimental code

## Description
* `mnist_cnn.py`
    * generate CNN trained to classify 28x28 greyscale images of singular digits 0-9
* `mnist_autoencoder.py`
    * generate MLP autoencoder for singular digits 0-9
* `mnist_generative_model.py`
    * adapted from the autoencoder, generates "new" number images from a 10x1 vector of arbitrary values


## Sample output

### mnist_autoencoder.py
* ![output_ae](results/images/autoencoder.gif)
* The video moves kind of fast, so some frames of interest explained:
    * Error correction: ![ec](results/images/ae_error_correction_0.png)
    * Standardization (removing horizontal 7 bar): ![s](results/images/ae_standardization_7.png)

### mnist_generative_model.py
* Transfer: Arbitrary inputs
    * ![output_gm_0](results/images/generative_model_integers_1.png)
* Retrain: Use inputs matching training data digits
    * Fixed Integers: ![output_gm_1](results/images/generative_model_integers_0.png)
    * Hybrid Animation: ![output_gm_1_mv](results/images/generative_transformation.gif)
