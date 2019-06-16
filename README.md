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
![output_ae](results/images/autoencoder.gif)

### mnist_generative_model.py
* Transfer: Arbitrary inputs
	* ![output_gm_0](results/images/generative_model_integers_1.png)
* Retrain: Use inputs matching training data digits
	* ![output_gm_1](results/images/generative_model_integers_0.png)
