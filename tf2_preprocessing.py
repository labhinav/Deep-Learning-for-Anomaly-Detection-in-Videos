import pathlib
import os
import sys

import tensorflow as tf
import numpy as np
import cv2
import skimage.transform


def segment_video(file_path, segment_len=16, output_shape=(128, 128)):
    '''Segments a video and writes into a TFRecord

    Each video is batched into frames of batch_size.
    All frames are converted to jpegs
    Returns: A list of all the segments in the video
    '''
    cap = cv2.VideoCapture(file_path)
    frame_count = cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)
    # ignore the remainder
    last_frame = int(frame_count // segment_len) * segment_len
    # stores frames of a segment
    temp = []
    # stores all segments of a video
    segments = []
    for frame_no in range(0, last_frame, segment_len):
        for count in range(segment_len):
            _, frame = cap.read()
            # antialiasing preserves artifacts when downscaling
            frame = skimage.transform.resize(frame,
                                             preserve_range=True,
                                             output_shape=output_shape,
                                             anti_aliasing=True)
            # encode as jpeg
            _, frame = cv2.imencode('.jpeg', frame)
            temp.append(frame)
        segments.append(temp)
    cap.release()
    return segments


def write_tf_record(file_path, output_path, segments):
    '''Writes a video as batches to a single TFRecord
    Args:
    filepath: filepath to original file
    segments: a list of segments of a video
    '''
    file_name = os.path.split(file_path)[-1]
    # removing extension
    file_name = os.path.splitext(file_name)[0]
    file_name += '.tfrecord'
    rel_path = os.path.join(output_path, file_name)

    # file to write to
    writer = tf.io.TFRecordWriter(rel_path)
    for itr in range(len(segments)):
        # creating record
        segment = tf.convert_to_tensor(segments[itr], dtype=tf.string)
        segment = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(segment).numpy()]
        ))
        name = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(file_name, 'utf-8')]
        ))
        num = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[itr]
        ))
        feature = {
            'segment': segment,
            'file': name,
            'num': num
        }
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        # writing to TFRecord
        writer.write(record.SerializetoString())

    writer.close()


if __name__ == "__main__":
    dir_path = "./Dataset_Samples"
    for file_name in os.listdir(dir_path):
        rel_path = os.path.join(dir_path, file_name)
        if os.path.isfile(rel_path):
            segments = segment_video(rel_path)
            write_tf_record(rel_path, './tfrecords', segments)
