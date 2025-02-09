import logging
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv1DTranspose,
    MaxPooling1D,
    BatchNormalization,
    Cropping1D,
    concatenate
)

import os
import librosa
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


import time
import logging

logging.basicConfig(
    filename="training.log",  # Log file name
    level=logging.INFO,       # Log everything from INFO level and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("STARTING...")
logging.info("MODULES IMPORTED...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logging.info(f"GPU detected: {gpus[0]}")
else:
    logging.info("No GPU detected. Please check your settings.")

# Random matrix multiplication for testing
with tf.device('/GPU:0'):  # Ensure operations are performed on the GPU
    a = tf.random.uniform([10000, 10000], dtype=tf.float32)
    b = tf.random.uniform([10000, 10000], dtype=tf.float32)
    
    start_time = time.time()
    c = tf.matmul(a, b)  # Matrix multiplication
    gpu_time = time.time() - start_time
    logging.info(f"GPU calculation completed in: {gpu_time:.4f} seconds")

logging.info("DATA LOADING SUCCESS ...")

segment_length = 88200 
batch_size = 10 
sampling_rate = 44100 

def crop(tensor, target_size,i):
    current_size = tensor.shape[1]
    crop_start = (current_size - target_size) // 2
    crop_end = current_size - target_size - crop_start
    cropping = (crop_start, crop_end)
    return Cropping1D(cropping=cropping,name=f'crop{i}')(tensor)

def U_net_build(input_shape):
    inputs = Input(shape=input_shape)
    i=0
    filters=206

    i+=1
    filters+=50
    x0=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(inputs)
    x0=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x0)
    x1 = BatchNormalization()(x0)
    x1=MaxPooling1D(pool_size=2)(x1)


    i+=1
    filters+=50
    x1=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x1)
    x1=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x1)
    x2 = BatchNormalization()(x1)
    x2=MaxPooling1D(pool_size=2)(x2)



    i+=1
    filters+=50
    x2=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x2)
    x2=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x2)
    x3 = BatchNormalization()(x2)
    x3=MaxPooling1D(pool_size=2)(x3)


    i+=1
    filters+=50
    x3=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x3)
    x3=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x3)
    x4 = BatchNormalization()(x3)
    x4=MaxPooling1D(pool_size=2)(x4)

    i+=1
    filters+=50
    x4=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x4)
    x4=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x4)
    x5 = BatchNormalization()(x4)
    x5=MaxPooling1D(pool_size=2)(x5)

    i+=1
    filters+=50
    x5=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x5)
    x5=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x5)
    x6 = BatchNormalization()(x5)
    x6=MaxPooling1D(pool_size=2)(x6)

    i+=1
    filters+=50
    x6=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x6)
    x6=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x6)
    x7 = BatchNormalization()(x6)
    x7=MaxPooling1D(pool_size=2)(x7)

    i+=1
    filters+=50
    x7=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x7)
    x7=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x7)
    x8 = BatchNormalization()(x7)
    x8=MaxPooling1D(pool_size=2)(x8)

    i+=1
    filters+=50
    x8=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}')(x8)
    x8=Conv1D(filters,15,activation='relu',padding='valid',name=f'{i}_cnc')(x8)
    x9 = BatchNormalization()(x8)
    x9=MaxPooling1D(pool_size=2)(x9)

    i+=1
    filters+=50
    bottleneck=Conv1D(filters,15,activation='relu',padding='valid',name='bottleneck')(x9)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(bottleneck)
    cropped_encoder_output = crop(x8, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x7, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x6, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x5, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x4, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x3, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x2, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x1, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    i-=1
    filters-=50
    x10=Conv1DTranspose(1,2,strides=2,activation='relu',padding='valid')(x10)
    cropped_encoder_output = crop(x0, x10.shape[1],i)
    x10 = concatenate([x10, cropped_encoder_output], axis=-1)
    
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)
    x10=Conv1D(filters,5,activation='relu',padding='valid')(x10)

    outputs= Conv1D(4,1,activation='tanh',padding='valid')(x10)
    model=Model(inputs,outputs)
    return model

input_shape = (88200, 1) 
model = U_net_build(input_shape)
model.summary()


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

 
# Constants
SAMPLE_RATE = 44100
CHUNK_SIZE = 88200  # 2 seconds
BATCH_SIZE = 8
segment_length = 62472

# Function to load and process audio
def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio
 
# Function to create chunks
def create_chunks_op(audio, chunk_size, typeof):
    if typeof == "mix":
        num_chunks = len(audio) // chunk_size
        audio = audio[:num_chunks * chunk_size]
        return np.split(audio, num_chunks)
    else:
        num_chunks = len(audio) // chunk_size
        audio = audio[:num_chunks * chunk_size]
        audio = np.split(audio, num_chunks)
        audio = [i[12864:chunk_size-12864] for i in audio]
        return audio

# Custom callback for progress bar
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=100, desc=f'Epoch {epoch+1}', 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        logging.info(f"BEGINNING EPOCH {epoch}")
    
    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)
    
    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        # Print metrics at the end of each epoch
        metrics_str = ' - '.join(f'{k}: {v:.4f}' for k, v in logs.items())
        print(f'\nEpoch {epoch+1} - {metrics_str}')

# Data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, musdb_path, chunk_size, batch_size):
        self.musdb_path = musdb_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.music_dirs = [os.path.join(musdb_path, d) for d in os.listdir(musdb_path) 
                          if os.path.isdir(os.path.join(musdb_path, d))]
        
    def __len__(self):
        return 100  # Number of batches per epoch
        
    def __getitem__(self, idx):
        inputs = []
        targets = []
        
        while len(inputs) < self.batch_size:
            music_dir = np.random.choice(self.music_dirs)
            try:
                # Load audio files
                mixture = load_audio(os.path.join(music_dir, "mixture.wav"))
                vocals = load_audio(os.path.join(music_dir, "vocals.wav"))
                others = load_audio(os.path.join(music_dir, "other.wav"))
                bass = load_audio(os.path.join(music_dir, "bass.wav"))
                drums = load_audio(os.path.join(music_dir, "drums.wav"))
                
                # Create chunks
                mixture_chunks = create_chunks_op(mixture, self.chunk_size, "mix")
                vocals_chunks = create_chunks_op(vocals, self.chunk_size, "op")
                drums_chunks = create_chunks_op(drums, self.chunk_size, "op")
                others_chunks = create_chunks_op(others, self.chunk_size, "op")
                bass_chunks = create_chunks_op(bass, self.chunk_size, "op")

                
                # Randomly select a chunk
                chunk_idx = np.random.randint(0, len(mixture_chunks))
                inputs.append(mixture_chunks[chunk_idx])

                comb = np.vstack((bass_chunks[chunk_idx], others_chunks[chunk_idx], drums_chunks[chunk_idx], vocals_chunks[chunk_idx])).T
                
                targets.append(comb)
                
            except Exception as e:
                print(f"Error processing {music_dir}: {str(e)}")
                continue
        
        # Convert to numpy arrays and add channel dimension
        batch_inputs = np.array(inputs)[:self.batch_size]
        batch_targets = np.array(targets)[:self.batch_size]
        
        return (np.expand_dims(batch_inputs, axis=-1),
                np.expand_dims(batch_targets, axis=-1))

musdb_train_path='/scratch/work/pallav/train/'
# Create data generator
logging.info("DATAGEN CREATED , NOW FITTING...")
train_gen = DataGenerator(musdb_train_path, CHUNK_SIZE, BATCH_SIZE)

# Create progress callback
progress_callback = ProgressCallback()
# Model training with progress bar
model.fit(
    train_gen,
    epochs=10,
    callbacks=[progress_callback],
    verbose=0  # Turn off default progress bar
)

try:
    model.save("modellop.h5")
except Exception as e:
    pass

try:
    model.save("modellop.keras")
except Exception as e:
    pass
    


