import numpy as np
import tensorflow as tf
import h5py
import os, sys

class Dataset:
    # Add max_samples=None to the constructor
    def __init__(self, data_path, split, input_length, data_desync=0, max_samples=None):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync
        self.max_samples_limit = max_samples # Store the limit

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'
        else:
            raise ValueError("Split must be 'train' or 'test'")

        # Determine the actual number of samples to load
        available_samples = corpus[split_key]['traces'].shape[0]
        if self.max_samples_limit is not None and self.max_samples_limit > 0:
            self.num_samples = min(available_samples, self.max_samples_limit)
        else:
            self.num_samples = available_samples # Load all if max_samples is None or invalid

        # --- Slice the data during loading ---
        # Load only the first `self.num_samples` rows
        self.traces = corpus[split_key]['traces'][:self.num_samples, :(self.input_length + self.data_desync)]
        self.labels = np.reshape(corpus[split_key]['labels'][:self.num_samples], [-1, 1])
        self.labels = self.labels.astype(np.int64)

        # --- Load and slice metadata ---
        # Load only the corresponding metadata rows
        metadata_subset = corpus[split_key]['metadata'][:self.num_samples]
        self.plaintexts = self.GetPlaintexts(metadata_subset)
        self.masks = self.GetMasks(metadata_subset)
        self.keys = self.GetKeys(metadata_subset)
        # --- End Slicing ---

        # --- Remove or adjust chunking for small datasets ---
        # Splitting might be unnecessary or cause errors if num_samples is small
        # For simplicity, we'll store them directly if num_samples is reasonably small
        # If you were loading millions, you might keep conditional chunking
        self.traces = [self.traces] # Store as a list containing one array
        self.labels = [self.labels] # Store as a list containing one array
        # --- End Chunking Adjustment ---

        corpus.close() # Close the HDF5 file

    # --- Metadata functions remain the same, but now receive the sliced metadata ---
    def GetPlaintexts(self, metadata_subset):
        plaintexts = []
        # Loop only up to the number of samples loaded
        for i in range(len(metadata_subset)):
            plaintexts.append(metadata_subset[i]['plaintext'][2])
        return np.array(plaintexts)

    def GetKeys(self, metadata_subset):
        keys = []
        # Loop only up to the number of samples loaded
        for i in range(len(metadata_subset)):
            keys.append(metadata_subset[i]['key'][2])
        return np.array(keys)

    def GetMasks(self, metadata_subset):
        masks = []
         # Loop only up to the number of samples loaded
        for i in range(len(metadata_subset)):
            masks.append(np.array(metadata_subset[i]['masks']))
        masks = np.stack(masks, axis=0)
        return masks
    # --- End Metadata modification ---

    def GetTFRecords(self, batch_size, training=False):
        # Adjusted to handle self.traces/self.labels being lists
        if not self.traces or not self.labels: # Check if lists are empty
             return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([]))) # Return empty dataset

        dataset = tf.data.Dataset.from_tensor_slices((self.traces[0], self.labels[0]))
        # Concatenate if there were multiple chunks (though we removed chunking for small N)
        for traces_chunk, labels_chunk in zip(self.traces[1:], self.labels[1:]):
            temp_dataset = tf.data.Dataset.from_tensor_slices((traces_chunk, labels_chunk))
            dataset = dataset.concatenate(temp_dataset) # Correctly concatenate

        def shift(x, max_desync):
            ds = tf.random.uniform([1], 0, max_desync+1, tf.dtypes.int32)
            ds = tf.concat([[0], ds], 0)
            # Ensure slicing bounds are valid even if trace length is smaller than expected
            slice_len = tf.minimum(self.input_length, tf.shape(x)[1] - ds[1])
            x = tf.slice(x, ds, [-1, slice_len])
            # Pad if necessary to ensure consistent shape (might happen with desync > 0)
            paddings = [[0, 0], [0, self.input_length - slice_len]]
            x = tf.pad(x, paddings)
            x.set_shape([None, self.input_length]) # Ensure shape consistency for TF graph
            return x

        if training:
            # Use self.num_samples which now reflects the potentially smaller dataset size
            dataset = dataset.shuffle(self.num_samples).repeat()
            if self.data_desync > 0 and self.input_length + self.data_desync <= self.traces[0].shape[1]:
                # Apply shift only if desync is enabled and original traces were long enough
                 dataset = dataset.batch(batch_size // 2 if batch_size > 1 else 1) # Avoid batch size 0
                 dataset = dataset.map(lambda x, y: (shift(x, self.data_desync), y), num_parallel_calls=tf.data.AUTOTUNE)
                 dataset = dataset.unbatch()
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
        else: # Evaluation
             if self.data_desync > 0 and self.input_length + self.data_desync <= self.traces[0].shape[1]:
                 # Apply deterministic shift (offset 0) if desync was used during loading
                 dataset = dataset.map(lambda x, y: (shift(x, 0), y), num_parallel_calls=tf.data.AUTOTUNE)

             dataset = dataset.batch(batch_size, drop_remainder=False) # Don't drop remainder for eval
             dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.prefetch(tf.data.AUTOTUNE)


    def GetDataset(self):
         # If you need the raw numpy arrays, concatenate the chunks back
        all_traces = np.concatenate(self.traces, axis=0) if self.traces else np.array([])
        all_labels = np.concatenate(self.labels, axis=0) if self.labels else np.array([])
        return all_traces, all_labels

# --- Example Usage (main function or separate script) ---
# Assuming you have the main script code elsewhere

# In your main script, when creating the Dataset objects:
# train_data = Dataset(data_path=FLAGS.data_path, split="train",
#                     input_length=FLAGS.input_length, data_desync=FLAGS.data_desync,
#                     max_samples=200) # Load only first 200 training

# test_data = Dataset(data_path=FLAGS.data_path, split="test",
#                    input_length=FLAGS.input_length, data_desync=FLAGS.data_desync,
#                    max_samples=50) # Load only first 50 test