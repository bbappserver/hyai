## Contributing
I do not activly monitor this repo. If you want to improve this software, then your change must not break existing behaviour in terms of learning archive delete, however you can extend the functionality and add better usability such as command line switches.  You may for exmaple give the user the ability to do multiple categories as long as you do not break binary archive-delete, or some flags to initialize the project or reset the caches.

Once you are satisfied with you change open a PR and then ping tzeentch on the  hydrus discord.

## Quickstart
Batteries are not included you will need to know a bit about python to use HYAI.
This program makes use of tensorflow.  You should have tesnsorflow setup on your machine to use your GPU.

A baseline model is supplied in `example-model` it is useless to you without training because our aesthetic prefences are different, but can be used as a baseline if you assume we have similar definitions of spammy trash. Simply extract it up one level into the root, otherwise a fresh model will be spawned for you.

1. Create two folders, one named `archive` and one named `trash`, and place them under a single directory `data`
2. Run `classify.py data` where `data` is the path of your samples this will generate the model checkpoint. If you don't know how a simple clasifier works read on.
3. Once the checkpoint is trained to your satisfaction. Create `settings.py` from `settings.example.py`, set fields as erequired/desired in `settings.py` then run `eval.py`.
4. Refine your checkpoint by providing new samples and repeating from step 2.

The target files will be classified if hyAI's confidence in the classifcation is within the target thresholds.  Classifcation is done by sending either `hyai:archive` or `hyai:trash` to the local `my tags` service.

The `eval.py` is not well optimized for all systems, it must downscale images to be of appropriate size to provide the the classifcation network.  You can work around this by having it use thumbnails instead of large files or changethe dataset pipeline to be less resource hungry on your system.

On first run the `classify.py` will preprocess your dataset to be a tensor of the size `224x224x3` as the start of the network requires this.  This basically involves scaling down and normalizing your images, which could take a while, and use a lot of resources.  Right now batch size is tuned to just be good enough to work on my machine.  You may need to shrink batch size if you have less VRAM available.

If you modify the contents of your example folders this caching step will need to happen again, and you **must delete the cache files** or tensorlfow will become confused about why there are more/fewer samples available than cached.

If you would like to conseserve disk space you can downscale your example images to `224x224`, as the network cannot make use of the larger images for training anyway.

Make sure tensorflow is set up to use your GPU if it is it will report that it is with something that looks like this.

```
I0000 00:00:1745388486.921321  190787 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4148 MB memory:  -> device: 0, name: NVIDIA CardName pci bus id: 0000:0c:00.0, compute capability: 7.5
Getting hashes from hydrus
Will process 5839421 hashes.
```

## How simple classifiers work

1.Identify features of the target
2.Create an embedding(set of labels) that corresponds to the 1 hot output vector
3.Train hyperparameters to corrspond the sample inputs to the output vector

If that's gibberish to you, and you actually care ypu can keep reading below,
but I haven't bothered to explain literally everything.  Here is a link to a video series that visually represents a similar situation to what we're doing here
as a simple written digit classifier.  You do not need to know the nitty gritty math to use neural nets but it will be helpful to visualize their shape.

[3blue1brown on neural nets](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

## How to raise accuracy of predictions

Like an infant that doesn't know a cookie from a cracker, or a puppy from a bear cub, the network must be instructed when it makes a mistake.  The easiest way to do this is to create a query that will fetch the items it is mistaken on and then feed these back into your input training data.  The queries to produce such data would be:

**Trash the network thought you would like**
Add to `data/trash` folder
```
File domain = trash
hyai:archive
```

**Good things the network thought were trash**
Add to `data/archive` folder
```
system:archive
hyai:trash
```

After extracting the target files train again and if the network starts performing better you can run eval again.

To bulk clear prior evaluations you can use the hydrus tag migration feature.  This is a powerful feature so be careful.  The serrings you want are

- my tags
- delete
- filter: `[hyai:archive,hyai:trash]` (uncheck the defautlt checked namespaced nad unnamspaced tags)
- mytags


## How it works
hyAI uses the mobilenet image classification network and transfer learning to tag your files as either `hyai:archive` or `hyai:trash`.  Mobile net is a neural network that converts a `244x244x3 float[0,1]` representation of an image into a number of **feature** numbers called the **latent space**.  These **features** are a mathmatical respresentation of the so called *latent varaibles* which are properties of the tags you might normally apply to an image, although not the tags themselves.  Like you know the difference between a person and a landscape but it might take you thousands of words to descriptivly say the difference.

The `mobilenet` comes with a so called **top** which is the end of the its architecture.  The top was used in its training to class 1024 different labels of images, so the top specifically labels objects like `gorilla`, `cat` or `swimmming pool`, but that's not enough to classify all possible images. So we remove the top and access the latents at the ned of the network directly.

The architecture summary is as follows

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ mobilenetv2_1.00_224 (Functional)    │ (None, 7, 7, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1024)                │       1,311,744 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1024)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │           1,025 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

```

`mobilenet` normally does a pooling layer to produce its last classifcation layer, we repeat this but we are interested in our custom labels.  These are still technically latent labels which don't really mean much to a human but are mathmatically useful.  They also do not necessarily encode full information about what is in the image, only the characteristics which are helpful in deciding whether trash or archive.

Next we have a dense layer this layer raises the "capacity" of our network which allows it to encode more combinations of latents.  For example you might normally not be into something in an image unless it is present with something else.

Then we have the droput layer.  A dropout randomly sets some hyper-parameters(the connections between layers) to zero.  Without dropout a training session can plateau into a case where the network makes all connections weighted to represent the same buckets in the previous layer, this keeps the error between the prediction and the truth low so the training optimizer will like to do this, but it means that the network will stop converging because it will saturate the capcity of the network with duplicates of the same weights instead of looking for new patterns in weightrs.

Finally we have another dense layer which simple connects the dropped out edges to a single output neuron.  This neuron represents the archive-deleteness of the input.  In out case we use `binary-crossentropy` as our **loss function** which means that our predeiction is a single real value between 0 and 1.  

Tensorflow uses alphabetical order to  determine which label should less than `0.5` and which should be more than `0.5`, so a value less than `0.5` indicates `archive` and a value greater than `0.5` indicates `trash`.
With this loss function the distance from `0.5` indicates confidence.  A value very close to `0` means high confidence of `archive` a value close to `1` indicates high confidence of trash.

In practice you must choose how to interpret the output of the network.  So you can choose an arbitrary cutoff threshold for how confident it must be to make a label.

For example you might say any value `<0.40` is `archive` and any value `>0.6` is `trash`, and any value in between is not certain enough to label.

### Labeling for more than 2 labels
You can make hyAI label for multiple categories by compiling the network with `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`, setting the capacity of the last node in the network to be the number of categories you are interested in, and providing multiple folders with labels.

The `eval.py` is also not setup for this scenario but you can find how to use this kind of evaluation in the tensorflow tutorial on transfer learning.

I previously experimented with this in the original implementation, so much of the code necessary to extract a label is present in both the comments of the python files.