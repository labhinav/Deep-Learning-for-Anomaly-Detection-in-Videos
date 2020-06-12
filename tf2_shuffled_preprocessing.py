import os
import sys
import threading
import logging
import queue
import glob
import random
import skimage.transform
import tensorflow as tf
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    # in order to import cv2 under python3
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
else:
    import cv2

# Logger for tracking progress
logging.basicConfig(format='%(asctime)s %(funcName)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)


class VideoFileStream():
    def __init__(self, file_name, index, segment_len=16, output_shape=(128, 128)):
        self.file_name = file_name
        self.index = index
        self.segment_len = segment_len
        self.output_shape = output_shape
        self.cap = cv2.VideoCapture(file_name)
        self.cur_frame = 0
        self.frame_count = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f'START:Reading:{self.file_name}'
                     f':total_frames:{self.frame_count}')

    def get_segment(self):
        # remainder frames will be dropped
        if self.has_segments():
            segment = self._make_segment()
            self.cur_frame += self.segment_len
            return segment
        else:
            logging.error(f'Called when no segments left:{self.file_name}')

    def _make_segment(self):
        segment = []
        for i in range(self.segment_len):
            status, frame = self.cap.read()
            if not status:
                logging.error(f'Couldnt get frame from {self.file_name}')
            frame = skimage.transform.resize(frame,
                                             preserve_range=True,
                                             output_shape=self.output_shape,
                                             anti_aliasing=True)
            _, frame = cv2.imencode('.jpeg', frame)
            segment.append(frame.tobytes())
        return segment

    def finish(self):
        self.cap.release()
        logging.info(f'DONE:Reading:{self.file_name}')

    def has_segments(self):
        return self.cur_frame + self.segment_len <= self.frame_count


def write_tf_record(output_path, file_num, segments):
    '''Writes a video as batches to a single TFRecord
    Args:
    filepath: filepath to original file
    output_path: Path of output dir
    segments: a list of segments of a video
    '''
    # Blank for now, will add back later
    file_name = ''
    out_file_name = "Data" + str(file_num) + '.tfrecord'
    rel_path = os.path.join(output_path, out_file_name)

    logging.debug(f'START:{rel_path}')
    # file to write to
    writer = tf.io.TFRecordWriter(rel_path)
    for itr in range(len(segments)):
        # creating record
        # Corresponding changes are to be made here
        # if the feature description in tf2_data_loader.py
        # is changed
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
        writer.write(record.SerializeToString())

    logging.debug(f'DONE:{rel_path}')
    writer.close()


def process_dir(dir_path, out_path, segment_len=16, output_shape=(128, 128),
                segments_per_file=500, parallel_files=950):
    file_list = glob.glob(os.path.join(dir_path, '*.mp4'))
    len_file_list = len(file_list)
    logging.info(f'{len_file_list} files in {dir_path}')
    videos = []
    for itr in range(min(len_file_list, parallel_files)):
        videos.append(VideoFileStream(file_list.pop(), itr,
                                      segment_len=segment_len,
                                      output_shape=output_shape))
    record_file_idx = 0
    segments = []
    while len(videos) != 0:
        rand_idx = random.randint(0, len(videos)-1)
        if videos[rand_idx].has_segments():
            segments.append(videos[rand_idx].get_segment())
        if not videos[rand_idx].has_segments():
            videos[-1], videos[rand_idx] = videos[rand_idx], videos[-1]
            videos[rand_idx].index = rand_idx
            videos[-1].finish()
            videos.pop()
            if len(file_list) > 0:
                videos.append(VideoFileStream(file_list.pop(), len(videos),
                                              segment_len=segment_len,
                                              output_shape=output_shape))
        if len(segments) == segments_per_file or len(videos) == 0:
            write_tf_record(out_path, record_file_idx, segments)
            record_file_idx += 1
            segments = []


if __name__ == "__main__":
    dir_path = './Dataset_Samples'
    out_path = './tfrecords'
    segment_len = 16
    segments_per_file = 500
    parallel_files = 950
    output_shape = (128, 128)
    file_list = glob.glob(os.path.join(dir_path, '*.mp4'))
    process_dir(dir_path, out_path, segment_len, output_shape,
                segments_per_file, parallel_files)
