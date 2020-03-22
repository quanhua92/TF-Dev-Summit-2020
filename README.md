# TF-Dev-Summit-2020
My Notes on [Tensorflow Dev Summit 2020](https://www.youtube.com/playlist?list=PLQY2H8rRoyvzuJw20FG82Lgm2SZjTdIXU)

# Table of Contents

- [Scaling Tensorflow data processing with tf.data](#scaling-tf-data)
- [TensorFlow 2 Performance Profiler](#profiler)
- [Research with TensorFlow](#research)
- [TensorFlow Hub: Making model discovery easy](#tf-hub)
- [Collaborative ML with TensorBoard.dev](#tensorboard-dev)
- [TF 2.x on Kaggle](#kaggle)
- [Learning to read with Tensorflow and Keras](#learn-to-read)
- [Making the most of Colab](#colab)

# Notes
<a id="scaling-tf-data"></a>

## Scaling Tensorflow data processing with tf.data

Link: [https://youtu.be/n7byMbl2VUQ](https://youtu.be/n7byMbl2VUQ )

### TLDR:

- Check official **tf.data** guide and performance guide.
- Reuse computation with **tf.data snapshot**
- Distribute computation with **tf.data service**

### Notes: 

Official **tf.data** guide: [https://www.tensorflow.org/guide/data](https://www.tensorflow.org/guide/data)

Normal pipeline:

```python
import tensorflow as tf

def expensive_preprocess(record):
	...
	
dataset = tf.data.TFRecordDataset(".../*.tfrecord")
dataset = dataset.map(expensive_preprocess)

dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size=32)

dataset = dataset.prefetch()

model = tf.keras.Model(...)
model.fit(dataset)
```

Improve single host performance guide: [https://www.tensorflow.org/guide/data_performance](https://www.tensorflow.org/guide/data_performance)

- Prefetch
- Parallel interleave
- Parallel map

Still bottleneck? 

**Idea 1: reuse computation with tf.data snapshot**

- Save the data transformation to hard disks and read later.

- Good for experimenting with model architectures, hyperparameter tuning.
- Should use before any transformation which requires randomness, such as cropping, shuffle.
- **Available in 2.3**

```python
def expensive_preprocess(record):
	...
	
dataset = tf.data.TFRecordDataset(".../*.tfrecord")
dataset = dataset.map(expensive_preprocess)

# snapshot all above
dataset = dataset.snapshot("/path/to/snapshot_dir")

dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size=32)

dataset = dataset.prefetch()
```



**Idea 2: distribute computation with tf.data service**

- Offload the computation to a cluster of workers
- Use **tf.data service** to scale horizontally: fault tolerant, exactly once guarantee
- **Available in 2.3**

```python
def randomized_preprocess(record):
	...
	
dataset = tf.data.TFRecordDataset(".../*.tfrecord")
dataset = dataset.map(randomized_preprocess) # need randomess so can not snapshot
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size=32)

# offload all above computations to a cluster of workers
dataset = dataset.distribute("<master_address>")

# below code will run on the host
dataset = dataset.prefetch()
```

<a id="profiler"></a>

## TensorFlow 2 Performance Profiler


Link: [https://youtu.be/pXHAQIhhMhI](https://youtu.be/pXHAQIhhMhI)

### TLDR:

- Use Profiler plugin in Tensorboard to aggregate & analyze the performance.
- Used extensively inside Google to tune products
- Tool Set: Overview, Input Pipeline Analyzer, TensorFlow Stats, Trace Viewer
- Each tool has recommendations for next step: link to other tools or tutorials

### Notes:

Tensorflow 2 Profiler Tool Set:

- Overview Page
- Input Pipeline Analyzer
- TensorFlow Stats
- Trace Viewer

Each tool has recommendations for next step: link to other tools or tutorials

Code:

```python
import tensorflow as tf

# Create your model & data preprocessing

# Create a TensorBoard callback
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="...",
                                            profile_batch='150, 160')
# 150, 160 means: do profile from batch 150 to 160

model.fit(...., callbacks=[tb_callback])
```

Overview Page

![Overview Page](images/profiler_overview.JPG)

Input Pipeline Analyzer

![Input Pipeline Analyzer](images/profiler_input_pipeline.JPG)

TensorFlow Stats

![TensorFlow Stats](images/profiler_stats.JPG)

Trace Viewer

![Trace Viewer](images/profiler_trace.JPG)



<a id="research"></a>

## Research with TensorFlow

Link: [https://youtu.be/51YtxSH-U3Y](https://youtu.be/51YtxSH-U3Y)

### TLDR:

- TensorFlow is **controllable, flexible, composable** -> useful for research.
- Use **tf.variable_creator_scope** to control the state of variables, layers, even with the existing kernels such as Keras layers.
- Use **tf.function(..., experimental_compile=True)** for automatic compilation
- Use **vectorization** for short, fast code
- Ragged data to work with non-tensor data types.

### Notes:

**Controlling State:**

- Can control the state of kernel inside TensorFlow library
- Tool: **tf.variable_creator_scope**
- Allow to do dependency injection to the available Tensorflow code & change behaviors instead of rewriting the function.

````python
# Custom variables
class FactorizedVariable(tf.Module):
    def __init__(self, a, b):
        self.a = a
        self.b = b
# how do I use above type as a step of computation: tf.matmul in this case
tf.register_tensor_conversion_function(FactorizedVariable, lambda x, *a, **k: tf.matmul(x.a, x.b))

# create the object inside the scope
def scope(next_creator, **kwargs):
    shape = kwargs["initial_value"]().shape
    # check if we want to create the variable or delegate for other as normal
    if len(shape) != 2: return next_creator(**kwargs)
    # use the custom variable
    return FactorizedVariable(tf.Variable(tf.random.normal([shape[0], 2])),
                              tf.Variable(tf.random.normal([2, shape[1]])))

# modify the Keras Dense Layer
with tf.variable_creator_scope(scope):
    d = tf.keras.layers.Dense(10)
    d(tf.zeros([20, 10]))
    
assert isinstance(d.kernel, FactorizedVariable)
````

**Compilation:**

- Compilation speedups with **tf.function(..., experimental_compile=True)**: do all magics to improve speed(avoid memory allocation, copy variables around, fuse everything into a single kernel, ...)

```python
# a custom kernel
def f(x):
    return tf.math.log(2 * tf.exp(tf.nn.relu(x+1)))

f = tf.function(f)
f(tf.zeros([100, 100])) # 0.007 ms

c_f = tf.function(f, experimental_compile=True)
c_f(tf.zeros([100, 100]))  # 0.005 ms --- ~25% faster
```

**Optimizers:**

- Keras makes it very easy to implement your own optimizers
- Keras optimizers + compilation = fast experimentation

Basic code to implement your own optimizers.

```python
class MyOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, ...):
        super().__init__(name="MyOptimizer")
        # initialize here
    def get_config(self): pass # rarely need to implement this
    def _create_slots(self, var_list):
        # create accumulators here
    def _resource_apply_dense(self, grad, var, apply_state=None):
        # apply gradients here
```

Using compilation for optimizers (for reference only): (has 2X speed up with 1 line of code)

```python
class MyOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, lr, power, avg):
        super().__init__(name="MyOptimizer")
        self.lrate, self.pow, self.avg = lr, power, avg
    def get_config(self): pass 
    def _create_slots(self, var_list):
        # create accumulators here
        for v in var_list: self.add_slot(v, "accum", tf.zeros_like(v))
    
    @tf.function(experimental_compile=True) # Add compilation 
    def _resource_apply_dense(self, grad, var, apply_state=None):
        # apply gradients here
        acc = self.get_slot(var, "accum")
        acc.assign(self.avg * tf.pow(grad, self.pow) + (1-self.avg)*acc)
        return var.assign_sub(self.lrate * grad / tf.pow(acc, self.pow))
```

**Vectorization:**

- Just write the elemental computation and TF takes care of batch operations.
- vectorization for short, fast code

***Example: Computing Jacobians***

- Approach 1: For Loop: it works

```python
x = tf.random.normal([10, 10])
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.exp(tf.matmul(x, x))
    
    j = []
    for i in range(x.shape[0]):
        jj = []
        for k in range(x.shape[1]):
            jj.append(t.gradient(y[i, k], x))
        j.append(jj)
    jacobian = tf.stack(j)
```

- Approach 2: Use **tf.vectorized_map**: more readability and performance boost 

```python
x = tf.random.normal([10, 10])
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.exp(tf.matmul(x, x))
    
    jacobian = tf.vectorized_map(
        lambda yi: tf.vectorized_map(
        	lambda yij: t.gradient(yij, x), yi), y)
```

- Approach 3: Use **jacobian**: best performance. However, use **tf.vectorized_map** if you want to write something custom.

```python
x = tf.random.normal([10, 10])
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.exp(tf.matmul(x, x))
    
    jacobian = t.jacobian(y, x)
```

**Ragged data to work with non-tensor data types.**

- Ragged Tensor manages non-tensor data types

***Example: Text Embeddings***

![Text Embeddings](images/research_embedding.JPG)

Ragged tensor: a representation for ragged data for efficient usage & use as a normal Tensor.

![Ragged Tensor](images/research_ragged_tensor.JPG)

```python
# Ragged data
data = [["this", "is", "a", "sentence"],
        ["another", "one"],
        ["a", "somewhat", "longer", "one", ",", "this"]]

# Ragged tensor
rt = tf.ragged.constant(data)
vocab = tf.lookup.StaticVocabularyTable(tf.lookup.KeyValueTensorInitializer(
["this", "is", "a", "sentence", "another", "one", "somewhat", "longer"],
tf.range(8, dtype=tf.int64)), 1)

rt = tf.ragged.map_flat_values(lambda x: vocab.lookup(x), rt)

# Get embedding table
embedding_table = tf.Variable(tf.random.normal([9, 10]))
rt = tf.gather(embedding_table, rt)

# Do computation on ragged tensor
tf.math.reduce_mean(rt, axis=1)
# Result has shape (3, 10) which is a Dense Tensor
```



<a id="tf-hub"></a>

## TensorFlow Hub: Making model discovery easy

Link: [https://www.youtube.com/watch?v=3seWxHGnDqM](https://www.youtube.com/watch?v=3seWxHGnDqM)

### TLDR:

- TensorFlow Hub: [https://tfhub.dev/](https://tfhub.dev/)
- Comprehensive collection of models: Image, Text, Video, Audio
- Pre-trained models ready for transfer learning & deployable anywhere you want
- How to publish to Tensorflow Hub: [Tutorial](https://github.com/tensorflow/hub/tree/master/tfhub_dev)

### Notes:

**Example: Style Transfer**

1. Find the model on tfhub: [https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
2. Use **tensorflow_hub** to download & run the model

````python
import tensorflow hub as hub

hub_handle = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
hub_module = hub.load(hub_handle)

stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

tensor_to_image(stylized_image)
````



**Example: Text classification**

Link: [tutorials/keras/text_classification_with_hub](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)

```python
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
.....
```

How to publish to Tensorflow Hub: [Tutorial](https://github.com/tensorflow/hub/tree/master/tfhub_dev)

<a id="tensorboard-dev"></a>

## Collaborative ML with TensorBoard.dev

Link: [https://youtu.be/v9a240kjAx4](https://youtu.be/v9a240kjAx4)

### TLDR:

- Enable collaborative ML by making it easy to share experiment results to [TensorBoard.dev](https://tensorboard.dev/)

### Notes:

1. Use TensorBoard as usually.

2. Upload to TensorBoard.dev & share the link

   ```bash
   tensorboard dev upload --logdir ./logs --name "My latest experiment" -- description "Simple comparison of several hyperparameters"
   ```

<a id="kaggle"></a>

## TF 2.x on Kaggle

Link: [https://youtu.be/IraU2xyAoKc](https://youtu.be/IraU2xyAoKc)

### TLDR:

- Kaggle has some new specific competition for TensorFlow: "TensorFlow 2.0 Question Answering" & "Flower Classification with TPUs"
- TF 2.1 + Kaggle == easy acceleration: no setup, not provisioning, free TPUs, GPUs

### Notes:

```python
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

# Detect hardward, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    tpu = None
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
    
with strategy.scope():
    model = tf.keras.Sequential([...])

model.compile(...)
```

<a id="learn-to-read"></a>

## Learning to read with TensorFlow and Keras

### TLDR:

- Build a sequence-to-sequence model with tensorflow_addons seq2seq to generate text
- Customize training by overriding the **train_step** method
- Hyper parameters tuning using **KerasTuner**
- TF ecosystem around text

### Notes:

Let's generate text!

- Get training data
- Preprocess text
- Train a model
- Generate text

**Load the data**

```python
lines = tf.data.TextLineDataset("train.txt")
```

**Clean the data**

```python
lines = lines.filter(lambda x: not tf.strings.regex_full_match(x, "_BOOK_TITLE_.*"))

punctuation = r'[!"#$%&()\*\+,-\./:;<=>?@[\\\]^_`{|}~\]'

lines = lines.map(lambda x: tf.strings.regex_replace(x, punctuation, ' '))
```

**Window the data**

```python
words = lines.map(tf.strings.split)
wordsets = words.unbatch().batch(11)
```

**Label the data**

```python
def get_example_label(row):
    example = tf.strings.reduce_join(row[:-1], separator=' ')
    example = tf.expand_dims(example, axis=0)
    label = row[-1:]
    return example, label

data = wordsets.map(get_example_label)
data = data.shuffle(1000)
```

**Preprocess the data using Preprocessing Layers**

- Easy data transformations
- Replace tf.keras.preprocessing
- Act like layers
- Serialized as part the model

```python
vocab_size = 5000
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size, output_sequence_length=10)

vectorize_layer.adapt(lines.batch(64))

vectorize_layer.get_vocabulary()[:5]
// [the and to a of]

vectorize_layer.get_vocabulary()[-5:]
// [jar isaac invented horrified herbs]
```

**Train a model**

Sequence-to-sequence learning with **tensorflow_addons**: **seq2seq 2.0**

````python
import tensorflow_addons as tfa

class EncoderDecoder(tf.keras.Model):
    def __init__(self, max_features=5000, embedding_dims=200, rnn_units=512):
        super().__init__()
        self.max_features = max_features
        self.vectorize_layer = TextVectorization(max_tokens=max_features, output_sequence_length=10)
        
        self.encoder_embedding = tf.keras.layers.Embedding(max_features + 1, embedding_dims)
        self.lstm_layer = tf.keras.layers.LSTM(rnn_unitss, return_state=True)
        
        self.decoder_embedding = tf.keras.layers.Embedding(max_features + 1, embedding_dims)

        projection_layer = tf.keras.layers.Dense(max_features)
        self.decoder = tfa.seq2seq.BasicDecoder(decoder_cell, samples,      output_layer=projection_layer)
        self.attention = tf.keras.layers.Attention()
        
    def train_step(self, data):
        x, y = data[0], data[1]
        x = self.vectorize_layer(x)
        # The vectorize layer pads; labels only need the first val
        y = self.vectorize_layer(y)[:, 0:1]
        y_one_hot = tf.one_hot(y, self.max_features)

        with tf.GradientTape() as tape:
            inputs = self.encoder_embedding(x)
            encoder_outputs, state_h, state_c = self.lstm_layer(inputs)

            attn_output = self.attention([encoder_outputs, state_h])
            attn_output = tf.expand_dims(attn_output, axis=1)

            targets = self.decoder_embedding(tf.zeros_like(y))
            concat_output = tf.concat([targets, attn_output], axis=-1)

            outputs, _, _ = self.decoder(concat_output, initial_state=[state_h, state_c])
            
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            self.compiled_metrics.update_state(y_one_hot, y_pred)
            return {m.name: m.result() for m in self.metrics}
````

**Configure training**

```python
model = EncoderDecoder()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(...),
    optimizer='adam',
    metrics=['accuracy'])
```

**Train**

```python
model.fit(data.batch(256), epochs=45, callbacks=    [tf.keras.callbacks.ModelCheckpoint('text_gen')])
```

**Hyperparameters tuning with KerasTuner**

- Distributable, Keras-native tuning
- Create and share tunable models
- [keras-team.github.io/keras-tuner](keras-team.github.io/keras-tuner)
- [Hyperparameter tuning with Keras Tuner](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html)

```python
import kerastuner as kt

def build_model(hp):
    model = EncoderDecoder(
    rnn_units=hp.Int('units', min_value=256, max_value=1100, step=256))
    
    model.compile(...)
    model.vectorize_layer.adap(lines.batch(256))
    return model

tuner = kt.tuners.RandomSearch(build_model, objective='accuracy', ...,  project_name='text_generation')

tuner.search(data.batch(256), epochs=45, callbacks=[tf.keras.callbacks.ModelCheckpoint('text_gen')])
```

**Predict the next word!**

```python
def predict_step(self, data, select_from_top_n=1):
    x = data
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    y_pred = tf.squeeze(outputs.rnn_output, axis=1)
    choices = tf.gather_nd(top_n, indices)
    words = [vectorize_layer.get_vocabulary()[i] for i in choices]
    return words
```

**Predict many words!**

```python
def predict(self, string_in, num_steps=50, select_from_top_n=1):
    s = tf.compat.as_bytes(string_in).split(b' ')
    for _ in range(num_steps):
        windowed = [b' '.join(s[-10:])]
        pred = self.predict_step([windowed], select_from_top_n=select_from_top_n)
        s.append(pred[0])
    return b' '.join(s)
```

Doing this at Google-scale

- tf.text
- KerasBert: cutting edge NLP model
- TFHub text modules



<a id="colab"></a>

## Making the most of Colab

### TLDR:

- Top 10 Colab Tricks for TensorFlow Users
- Introduce to Colab Pro for 10$ / month: faster GPUs, longer runtimes, more memory
- The free version of Colab is not going away

### Notes:

How does Colab work?

- Pre-warmed VMs
- Pre-installed packages
- Resource limits

Top 10 Colab Tricks for TensorFlow Users

**#10: Specify TensorFlow version**

```python
%tensorflow_version 2.x

# 1.x
%tensorflow_version 1.x
```

**#9: Use TensorBoard right in Colab**

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

**#8: TFLite? No problem!**

Train in Colab -> Deploy to mobile

**#7: Use TPUs**

Runtime -> Change runtime type

**#6: Use local runtimes**

Connect -> Connect to local runtime...

**#5: The Colab scratchpad**

Link: [https://colab.research.google.com/notebooks/empty.ipynb](https://colab.research.google.com/notebooks/empty.ipynb)

**#4: Copy data to Colab VMs**

This often results in speedup

**#3: Mind your memory**

The best thing is to not run out of memory at all and mind your memory when you're doing your work to avoid resource limits

**#2: Close tabs when done**

This will help you disconnect to VMs sooner to avoid resource limits

**#1: Only use GPUs when needed**

When you're doing works that doesn't need GPU, just use default runtime CPU and use GPU later.



Introduce to Colab Pro for 10$ / month: faster GPUs, longer runtimes, more memory



What's next in Colab:

- The free version of Colab is not going away
- Send feedback via COlab
- @googlecolab on Twitter