# Functions for dataset processing

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float64')

def parse_records_wrapper(branch_sensors, dim_y, dim_out, include_weights=False):
    def parse_records(example_proto):
        features = { 
            'Xf': tf.io.FixedLenFeature([branch_sensors], tf.float32),
            'Xp': tf.io.FixedLenFeature([dim_y], tf.float32),
            'Y':  tf.io.FixedLenFeature([dim_out], tf.float32),
        }   
        if include_weights:
            features['W'] = tf.io.FixedLenFeature([], tf.float32)

        parsed_features = tf.io.parse_single_example(example_proto, features)

        if include_weights:
            return ((parsed_features['Xf'], parsed_features['Xp']),
                    parsed_features['Y'],
                    parsed_features['W'])
        else:
            return ((parsed_features['Xf'], parsed_features['Xp']),
                    parsed_features['Y'])

    return parse_records


AUTOTUNE = tf.data.experimental.AUTOTUNE
def load_dataset(filepaths, branch_sensors, dim_y, dim_out, batch_size,
                 use_weights=True,
                 preads=1,
                 shuffle_buffer=0):
    # Read records
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=preads)

    # Disable order
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(ignore_order)

    # Parse using parse_records_wrapper
    records_parser = parse_records_wrapper(branch_sensors, dim_y, dim_out, include_weights=use_weights)
    dataset = dataset.map(records_parser, num_parallel_calls=AUTOTUNE)

    # Shuffle, prefetch, and batch
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset

