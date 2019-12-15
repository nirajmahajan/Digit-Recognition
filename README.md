# Digit Classifier

A simple digit classifier trained in **PyTorch** using the MNIST database. This classifier is built on several models.

1. [Softmax Classifier](https://github.com/nirajmahajan/Digit-Recognition/models/softmax) : Accuracy of 92.47%
2. [1-Hidden Layer Neural Network](https://github.com/nirajmahajan/Digit-Recognition/models/NN) : Accuracy of 95.77%
3. [Deep Neural Network](https://github.com/nirajmahajan/Digit-Recognition/models/DNN) : Accuracy of 96.54%
4. [Convolutional Neural Network](https://github.com/nirajmahajan/Digit-Recognition/models/CNN) : Accuracy of 98.38% 

## Code Structure:

The code structure for all the models is quite similar. Each model has the following files:

- **helpers.py** : Contains all essential imports and functions
- **train.py** : Used for training and testing the accuracy of a model
- **predict.py** : Used for predicting the digit in a given image
- **model/** : Has a pre-trained model

Apart from this, the MNIST data and a few sample images are located in the [Utils](https://github.com/nirajmahajan/Digit-Recognition/utils) folder.

## Usage of Code:

1. To train any model:

   ```bash
   python3 train.py # Note that this will replace the pre existing model
   ```

2. To check the accuracy of any model (present in the 'model' directory):

   ```bash
   python3 train.py -use_trained_model
   ```

3. To predict the digit from an image:

   ```bash
   python3 predict.py --path <Path To Image>
   ```

