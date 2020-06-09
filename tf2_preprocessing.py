import pathlib
import os
import sys

import tensorflow as tf
import numpy as np
import cv2


def segment_video(filename, batch_size):
    '''Segments a video and writes into a TFRecord

    Each video is batched into frames of batch_size and written to a single
    TFrecord
    '''
    cap = cv2.VideoCapture(filename)
    frame_count = cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count


if __name__ == "__main__":
    dir_path = "./Dataset_Samples"
    total = 0
    for file_name in os.listdir(dir_path):
        rel_path = os.path.join(dir_path, file_name)
        if os.path.isfile(rel_path):
            total += segment_video(rel_path, 1)
    print(total)
