## Keras
Keras is a high-level deep learning API that makes building neural networks simpler and more readable.

Keras provides built-in datasets: `tf.keras.datasets.cifar10`
Keras provides building blocks (Conv2D, Dense, Dropout, etc.): `tf.keras.layers`
Keras. provides the model: `tf.keras.models.Model`

While TensorFlow is the powerful engine doing the heavy math (matrix multiplications, gradients, GPU ops), Keras is the friendly interface on top that allows building models in clean, readable Python instead of raw TensorFlow operations.

For example, without Keras, to do the convolution operation, all these definitions have to be set up manually: weight matrices, write backpropagation, manage tensor shapes, etc. With Keras, it is done with just one line:
`x = Conv2D(32, (3, 3), activation='relu')(x)`

Read details about Conv2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

**GPU ops**
[GPU](GPU.md)

**Model**
[Model](Model.md)

**Back to Main Note**
[MainNote](MainNote.md)



