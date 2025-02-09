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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import time
import logging

logging.basicConfig(
    filename="continued_training.log",  # Log file name
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


for dirname, _, filenames in os.walk('/scratch/work/pallav'):
    for filename in filenames:
        os.path.join(dirname, filename)

list1=[]
for dirname, _, filenames in os.walk('/scratch/work/pallav/train'):
    for filename in filenames:
        list1.append(os.path.join(dirname, filename))
data = []
feature_per_song=5
for i in range(0,len(list1),feature_per_song):
    row=list1[i:i+feature_per_song]
    data.append(row + [f"{i // feature_per_song + 1}"])
df=pd.DataFrame(data)
print(df.iloc[9][4])
df.columns=['drums','vocals','bass','others','mixture','index']

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

input_shape = (88200, 1) 
model = load_model("modellop.h5")
model.summary()
logging.info("MODEL LOADED SUCCESSFULLY")

new_learning_rate = 0.0003  # Reduced from default 0.001
model.compile(
    optimizer=Adam(learning_rate=new_learning_rate),
    loss='mean_squared_error',
    metrics=['mae']
)


import os
import librosa
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
 
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
        if(epoch%50):
            try:
                model.save(f"ContinuedModel{epoch}.h5")
                logging.info(f"ModelSaved ContinuedModel{epoch}.h5 Success")
            except Exception as e:
                logging.info(f"ModelSavedFailed ContinuedModel{epoch}.h5 NotSuccess")
                pass
            

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
    epochs=200,
    callbacks=[progress_callback],
    verbose=0  # Turn off default progress bar
)

try:
    model.save("FinalModel300epoch.h5")
    logging.info(f"FinalModelSaved FinalModel300epoch.h5 Success")
except Exception as e:
    logging.info(f"FinalModelSavedFail FinalModel300epoch.h5 Failed")
    pass

try:
    model.save("FinalModel300epoch.keras")
    logging.info(f"FinalModelSaved FinalModel300epoch.keras Success")
except Exception as e:
    logging.info(f"FinalModelSavedFail FinalModel300epoch.keras Failed")
    pass
    


