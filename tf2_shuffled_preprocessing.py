import os
import sys
import threading
import concurrent.futures
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
    '''Used to encapsulate video streams and maintain state
    '''
    def __init__(self, file_name, index, segment_len=16, output_shape=(128, 128)):
        '''
        Args:
        file_name: path of file to open as video stream
        index: Used to randomly select a video for reading
        segment_len: #frames per segment
        output_shape: shape of jpeg images of each frame
        '''
        self.file_name = file_name
        self.index = index
        self.segment_len = segment_len
        self.output_shape = output_shape
        self.cap = cv2.VideoCapture(file_name)
        # To maintain how much is read
        self.cur_frame = 0
        self.frame_count = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        # Tracking progress
        logging.info(f'START:Reading:{self.file_name}'
                     f':total_frames:{self.frame_count}')

    def get_segment(self):
        '''Checks for and returns the next segment of self.segment_len frames
        Each segment is a list of byte strings
        each of which is an encoded jpeg
        '''
        # remainder frames will be dropped
        if self.has_segments():
            segment = self._make_segment()
            self.cur_frame += self.segment_len
            return segment
        else:
            logging.error(f'Called when no segments left:{self.file_name}')

    def _make_segment(self):
        '''Returns the next segment of self.segment_len frames
        '''
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
        '''Clean up by closing the video stream
        '''
        self.cap.release()
        logging.info(f'DONE:Reading:{self.file_name}')

    def has_segments(self):
        '''Returns true if there are more segments available
        '''
        # Ignores remainder frames
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
    '''Processes all the mp4 files in dir_path and outputs them as TFRecord files
    Args:
    dir_path: path of dir containing the input videos
    out_path: path of dir to output TFRecords to
    segment_len: #frames in each segment
    output_shape: dimensions of the processed frame
    segments_per_file: #segments in each TFRecord
    parallel_files: #files to simultaneously read from
    '''
    # For multi threading writes
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3,
                                                     thread_name_prefix='writer')
    # get all file paths to .mp4 files
    file_list = glob.glob(os.path.join(dir_path, '*.mp4'))
    len_file_list = len(file_list)
    logging.info(f'{len_file_list} files in {dir_path}')
    videos = []
    for itr in range(min(len_file_list, parallel_files)):
        videos.append(VideoFileStream(file_list.pop(), itr,
                                      segment_len=segment_len,
                                      output_shape=output_shape))
    # serial num of the next TFRecord to write to
    record_file_idx = 0
    # to store are segments of a TFRecord
    segments = []
    while len(videos) != 0:
        # randomly pick a video to get a frame from
        rand_idx = random.randint(0, len(videos)-1)
        # get the segment, if any
        if videos[rand_idx].has_segments():
            segments.append(videos[rand_idx].get_segment())
        # If out of segments, replace with a new video
        if not videos[rand_idx].has_segments():
            # swap with last for fast delete on avg
            videos[-1], videos[rand_idx] = videos[rand_idx], videos[-1]
            videos[rand_idx].index = rand_idx
            videos[-1].finish()
            videos.pop()
            # load a new video if available
            if len(file_list) > 0:
                videos.append(VideoFileStream(file_list.pop(), len(videos),
                                              segment_len=segment_len,
                                              output_shape=output_shape))
        # write the TFRecord
        if len(segments) == segments_per_file or len(videos) == 0:
            # Multi threaded the write and continue
            executor.submit(write_tf_record, out_path,
                            record_file_idx, segments)
            record_file_idx += 1
            segments = []
    # Wait for all writes to complete
    executor.shutdown()


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
