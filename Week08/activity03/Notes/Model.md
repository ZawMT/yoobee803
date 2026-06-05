## Model
A model is essentially a mathematical function that takes an input and produces an output.
For example:
Input ⇒ an image (32×32 pixels)
Output ⇒ a guess ("that's a cat")

It learns this function by looking at thousands of examples (training data) and gradually adjusting itself to get better at guessing.

The model is made up of **layers** — each layer does a small transformation on the data and passes it to the next:
Image ⇒ [Conv2D] ⇒ [BatchNorm] ⇒ [MaxPool] ⇒ ... ⇒ [Dense] → "cat"

Each layer has weights (numbers) that get tuned during training. That's what learning actually is — adjusting millions of numbers until the output is correct.

**At the start**
- The model is basically random — weights are initialised with random numbers
- It knows nothing — if you asked it to classify an image right now, it would just guess randomly

** During training (the "looking and learning" part)**
- Show it an image → it makes a guess
- Compare the guess to the correct answer → measure how wrong it was (this is the loss)
- Adjust the weights slightly to be less wrong next time
- Repeat 50,000 × 5 epochs = millions of times 

**At the end**
- All those small adjustments have accumulated into a set of weights that works
- That final set of weights is the function it "created"

When a model is saved, e.g. `model.save('CIFAR_10_tens.h5')`, it is essentially saving the function it discovered — those millions of tuned numbers — so there is no need to rediscover it again next time.

**Back to Main Note**
[MainNote](MainNote.md)