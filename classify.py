import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, RandomFlip,Dropout,RandomRotation,RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import random
import os
#import matplotlib.pyplot as plt
#import subprocess
#import re

#Suppress image load warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
K.clear_session()

looping=True
#you can ignore this I like to use a reduced dataset to get the overall form of the fit.
#before trying to train agianst the full set
redux=False 

training_set_dir='./data'

directory_path = training_set_dir if not redux else f'{training_set_dir}-redux' # Dataset path
topless=True
checkpoint_path= 'best_model.topless.keras' if topless else 'best_model.keras'
preprocess_data='cache/dat.cache' if not redux else 'redux/dat.cache'
preprocess_validate='cache/val.cache' if not redux else 'redux/val.cache'
image_size = (224, 224)  # MobileNetV2 input size
#batch_size = 3*640  # Max for 32GiB RAM very stable gradient estimate
#batch_size = 64  # Less stable gradient estimate but faster epochs.
image_load_batch=2*512
prefetch_size=16*image_load_batch
validate_load_batch=int(image_load_batch*7/3)
parallel_image_processors=2*64
batch_size=90
processed_batch_size=2968
prefetch=tf.data.AUTOTUNE
validation_split = 0.3  # 70/30 train/validation split
max_epochs = 100  # Max epochs, stops early if accuracy reached
target_val_accuracy = 0.9999  # Target: 1 error in 10000
valfreq=1#validate only ever n-th epoch

seed=random.randint(0,9999999)


# Load training and validation datasets from directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory_path,
    validation_split=validation_split,
    subset='training',
    seed=seed,
    image_size=image_size,
    batch_size=image_load_batch,
)
# train_dsb=train_ds
# step_per_epoch=len(train_ds)//image_load_batch
# max_epochs= min(max_epochs, len(train_ds)//image_load_batch)
# i=1



#repeat data if you don't have enoug to fill the target
#this is usually an indication that the choice of batch is too large for the choice of epochs
# while max_epochs <=0:
#     train_ds=train_dsb.repeat(i)
#     max_epochs= len(train_ds)//image_load_batch
#     i+=1

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory_path,
    validation_split=validation_split,
    subset='validation',
    seed=seed,
    image_size=image_size,
    batch_size=validate_load_batch
)

# Do I/O while training is in progress
train_ds = train_ds.prefetch(prefetch)
val_ds = val_ds.prefetch(prefetch)

# Preprocess images for MobileNetV2 (Optimization: parallelized on CPU)
def preprocess(image, label):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

#flip and zoom preprocessing to amplify samples
aug = Sequential([
    RandomFlip(),
    RandomZoom(0.1)
    ])

def augment(image, label):
    return aug(image),label

#Cache images processed for correct input and then stream them to the next batch.
train_ds = train_ds.map(preprocess, num_parallel_calls=parallel_image_processors).cache(filename=preprocess_data).shuffle(4096).map(augment).prefetch(prefetch_size)
val_ds = val_ds.map(preprocess, num_parallel_calls=parallel_image_processors).cache(filename=preprocess_validate).map(augment).prefetch(prefetch_size)

def build_model(topless=True):

    if topless:
        # Build MobileNetV2 model with frozen base and custom head
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze base 
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(), #important feature extractor
            Dense(1024,activation='relu'), #convert features into combos
            Dropout(0.003), #randomly set some combo to bucket contributions to 0 to prevent overfitting (enough should still be left to influence a bucket)
            Dense(2, activation='sigmoid')  # Binary output
        ])
    else:
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
        base_model.trainable = False  # Freeze base t
        model = Sequential([
            base_model,
            Dense(64,activation='sigmoid'), #generate combination contributions, and overprovision so knockout doens't dominate
            Dropout(0.006), #shuffle contribution factor
            Dense(1, activation='sigmoid')  # Binary output
        ])

    return model



#Preparation done now do training
base_lr=0.0133 #start with a broad exploration of the hyperparametar space

while True:
    # Load existing checkpoint or build new model
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        print("No checkpoint found, building new model")
        model = build_model(topless=topless)

    #If you would prefer to do multiple categories change this
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss='binary_crossentropy'

    # Compile model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
                loss=loss,
                metrics=['accuracy'])

    # Custom callback for accuracy target and learning rate reduction
    class RecoveryAndTargetCallback(Callback):
        def __init__(self, patience=10, min_lr=1e-6, factor=0.8, target_accuracy=0.999):
            super().__init__()
            self.patience = patience
            self.min_lr = min_lr
            self.factor = factor
            self.target_accuracy = target_accuracy
            self.best_val_loss = float('inf')
            self.epochs_without_improvement = 0

        def on_epoch_end(self, epoch, logs=None):
            current_val_loss = logs.get('val_loss')
            current_val_accuracy = logs.get('val_accuracy')
            current_accuracy = logs.get('accuracy')

            #if we can't match the test set but somehow luck into validation accuracy, that's not really accurate
            #but the ability to guess what you haven't seen before is more important weight both accordingly
            weighted_accuracy= ( (0.8*current_val_accuracy) + (0.2*current_accuracy))
            logs['w_accuracy']=weighted_accuracy
            if current_val_loss is None: 
                return

            if weighted_accuracy is not None and weighted_accuracy > self.target_accuracy:
                print(f"Stopping because weighted accuracy {weighted_accuracy} is better than target {self.target_accuracy} ")
                self.model.stop_training=True
                global looping
                looping=False
                return

            if epoch>40 and current_val_accuracy< 0.68:
                #somehow val accuracy is quite low this is usually an indication
                #that we are overfitting the test data so stop this iteration (and reload the checkpoint)
                print("excessive validation error, stopping run")
                self.model.stop_training=True

            # Reduce learning rate if stalled (Optimization: escape plateaus)
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    current_lr = float(self.model.optimizer.learning_rate)
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    print(f"\nNo improvement in validation loss for {self.patience} epochs. "
                        f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}.")
                    self.model.optimizer.learning_rate = new_lr
                    self.epochs_without_improvement = 0

    # Define callbacks for checkpointing and recovery
    recovery_callback = RecoveryAndTargetCallback(patience=80, min_lr=1e-6, 
    factor=0.9, target_accuracy=target_val_accuracy)

    #save only the model which performed best on items it had not seen before
    model_checkpoint = ModelCheckpoint(f'{checkpoint_path}', monitor='w_accuracy', save_best_only=True, mode='max')

    # Train model
    history = model.fit(
        train_ds,
        epochs=max_epochs,
        validation_data=val_ds,
        validation_freq=valfreq,
        callbacks=[recovery_callback, model_checkpoint]
    )

    if not looping: break


