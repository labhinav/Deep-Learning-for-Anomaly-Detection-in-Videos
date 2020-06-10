import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

# Corresponding changes are to be made here
# if the feature description in tf2_preprocessing.py
# is changed
feature_description = {
    'segment': tf.io.FixedLenFeature([], tf.string),
    'file': tf.io.FixedLenFeature([], tf.string),
    'num': tf.io.FixedLenFeature([], tf.int64)
}


def build_dataset(dir_path, batch_size=16, file_buffer=500*1024*1024,
                  shuffle_buffer=1024, label=1):
    '''Return a tf.data.Dataset based on all TFRecords in dir_path
    Args:
    dir_path: path to directory containing the TFRecords
    batch_size: size of batch ie #training examples per element of the dataset
    file_buffer: for TFRecords, size in bytes
    shuffle_buffer: #examples to buffer while shuffling
    label: target label for the example
    '''
    # glob pattern for files
    file_pattern = os.path.join(dir_path, '*.tfrecord')
    # stores shuffled filenames
    file_ds = tf.data.Dataset.list_files(file_pattern)
    # read from multiple files in parallel
    ds = tf.data.TFRecordDataset(file_ds,
                                 num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                 buffer_size=file_buffer)
    # randomly draw examples from the shuffle buffer
    ds = ds.shuffle(buffer_size=shuffle_buffer,
                    reshuffle_each_iteration=True)
    # batch the examples
    # dropping remainder for now, trouble when parsing - adding labels
    ds = ds.batch(batch_size, drop_remainder=True)
    # parse the records into the correct types
    ds = ds.map(lambda x: _my_parser(x, label, batch_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def _my_parser(examples, label, batch_size):
    '''Parses a batch of serialised tf.train.Example(s)
    Args:
    example: a batch serialised tf.train.Example(s)
    Returns:
    a tuple (segment, label)
    where segment is a tensor of shape (#in_batch, #frames, h, w, #channels)
    '''
    # ex will be a tensor of serialised tensors
    ex = tf.io.parse_example(examples, features=feature_description)
    ex['segment'] = tf.map_fn(lambda x: _parse_segment(x),
                              ex['segment'], dtype=tf.uint8)
    # ignoring filename and segment num for now
    # returns a tuple (tensor1, tensor2)
    # tensor1 is a batch of segments, tensor2 is the corresponding labels
    return (ex['segment'], tf.fill((batch_size, 1), label))


def _parse_segment(segment):
    '''Parses a segment and returns it as a tensor
    A segment is a serialised tensor of a number of encoded jpegs
    '''
    # now a tensor of encoded jpegs
    parsed = tf.io.parse_tensor(segment, out_type=tf.string)
    # now a tensor of shape (#frames, h, w, #channels)
    parsed = tf.map_fn(lambda y: tf.io.decode_jpeg(y), parsed, dtype=tf.uint8)
    return parsed


def display_segment(segment, batch_size):
    fig = plt.figure(figsize=(16, 16))
    columns = int(math.sqrt(batch_size))
    rows = math.ceil(batch_size / float(columns))
    for i in range(1, columns*rows + 1):
        img = segment[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    dir_path = './tfrecords'
    batch_size = 16
    ds = build_dataset(dir_path, batch_size=batch_size)
    for batch in ds:
        for segment in batch[0]:
            display_segment(segment, batch_size)
            print('Close the plot window manually')
            inp = input("Hit q to quit, any other key to continue: ")
            if inp == 'q':
                break
        if inp == 'q':
            break
