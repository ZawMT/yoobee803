## Understanding the codes
**Cell 0**
 Imports all required libraries. 
 TensorFlow for building/training the model, NumPy for array operations, Matplotlib for plotting.

**Cell 1**
Downloads the CIFAR-10 dataset and splits it into training (50,000 images) and test (10,000 images) sets. Each image is 32×32 pixels with 3 colour channels (RGB).

About `keras`: [Keras](Keras.md)

**Cell 2**
Data preprocessing.
Divides pixel values by 255 to scale them from [0–255] to [0.0–1.0] — neural networks train better with small normalised values
Flattens labels from shape (50000, 1) to (50000,) — required by sparse_categorical_crossentropy.

**Cell 3**
Visualisation
Plots a 5×5 grid of the first 25 training images so you can see what the data looks like before training.

**Cell 4**
Building model
Builds a CNN with 3 blocks, each block doing:
- Two Conv2D layers to detect features (edges, shapes, textures)
- BatchNormalization to stabilise and speed up training
- MaxPooling2D to downsample and reduce spatial size

Then flattens into a fully connected section:
- Dropout(0.2) to prevent overfitting
- Dense(1024) hidden layer
- Dense(10, softmax) output layer — one score per class, probabilities sum to 1

About Model: [Model](Model.md)

**Cell 5**
Compilation
```
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
```
 Configures how the model learns:
- adam — adaptive learning rate optimiser, good default choice
- sparse_categorical_crossentropy — standard loss for multi-class classification with integer labels
- accuracy — metric to track during training 

**Cell 6**
Training
`r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)`
Trains the model for 5 passes over the training data. After each epoch it evaluates on the test set to track how well it generalises. 
__This cell takes time.__

**Cell 7**
Train with Data Augmentation
```
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
      width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
```
Tries to artificially expand the training set by randomly shifting and flipping images on the fly — a common technique to improve generalisation and reduce overfitting. 
__This cell takes time.__

**Cell 8**
Plot accuracy
```
plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
```
Plots training vs validation accuracy over epochs. If the gap between them is large, the model is overfitting.

**Cell 9**
Predict a Single Image

`predicted_label = labels[model.predict(p).argmax()]`
Takes one test image, reshapes it to (1, 32, 32, 3) (model expects a batch), runs it through the model, and picks the class

```
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
      width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
```
Tries to artificially expand the training set by randomly shifting and flipping images on the fly — a common technique to improve generalisation and reduce overfitting. 

**Cell 10**
Saving the model