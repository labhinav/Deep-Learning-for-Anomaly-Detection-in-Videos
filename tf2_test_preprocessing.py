import os
import sys
import threading
import logging
import queue
import concurrent.futures
import pandas as pd
from sklearn.model_selection import train_test_split
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

    def __init__(self, file_path, timestamps, segment_len=16,
                 output_shape=(128, 128)):
        '''
        Args:
        file_name: path of file to open as video stream
        timestamps: A list of alternating start, stop times of the anomalies
                   -1 if no anomaly. len(timestamp) = 4
        segment_len: #frames per segment
        output_shape: shape of jpeg images of each frame
        '''
        self.file_name = os.path.split(file_path)[-1]
        self.segment_len = segment_len
        self.output_shape = output_shape
        self.timestamps = timestamps
        self.cap = cv2.VideoCapture(file_path)
        # Check is corrupt
        if not self.cap.isOpened():
            logging.fatal(f'{file_path} is CORRUPT. Delete it and retry.')
            raise Exception('Corrupt File')
        self.frame_count = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        # 1-based indexing
        # index of next start frame
        self.frame_no = 1
        # remainder frames will be dropped
        self.segments_left = self.frame_count // self.segment_len
        logging.info(f'START:Reading:{file_path}\
                     :total_frames:{self.frame_count}')

    def dec_segments_left(self):
        '''Decrements the #segments_left by 1
        '''
        self.segments_left -= 1

    def finish(self):
        '''Clean up by closing the video stream
        '''
        self.cap.release()
        logging.info(f'DONE:Reading:{self.file_name}')

    def make_segment(self):
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
        num = self.frame_no
        label = self.get_label()
        self.frame_no += self.segment_len
        return {'segment': segment,
                'num': num,
                'file': self.file_name,
                'label': label}

    def get_label(self):
        '''Returns the label for the segment based on the start frame
        1 - anomalous
        0 - normal
        '''
        label = False
        # Check with first occurence of anomaly
        if self.timestamps[0] != -1:
            if not(self.frame_no > self.timestamps[1] or
                   self.frame_no + self.segment_len <= self.timestamps[0]):
                label = True
        # Check with second occurence of anomaly
        if self.timestamps[2] != -1:
            if not(self.frame_no > self.timestamps[3] or
                   self.frame_no + self.segment_len <= self.timestamps[2]):
                label = True
        return label

    def get_all_segments(self, segments_q):
        '''Gets all the segments in the video and
        puts it in segments_q
        Args:
        segments_q: queue.Queue to put segments in
        '''
        while self.segments_left > 0:
            self.dec_segments_left()
            segment = self.make_segment()
            segments_q.put(segment)
        self.finish()


def read_segments(data_frame, base_path, segments_q,
                  segment_len=16, output_shape=(128, 128),
                  thread_limit=32):
    '''Processes all the mp4 files in dir_path TFRecord and stores the segments in queue

    Args:
    data_frame: DataFrame containing all the videos to process along with timestamps
    base_path: path to the dir with the dirs for each class
    segments_q: queue.Queue to put segments into
    segment_len: #frames in each segment
    output_shape: dimensions of the processed frame
    thread_limt: max #threads to use for reading
    '''
    # get paths to all mp4 s
    logging.info(f'Processing {len(data_frame)} files')
    # manages the threads for reads
    reader_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_limit)
    for row in data_frame.iter_tuples():
        file_path = os.path.join(base_path, row[2], row[1])
        timestamps = [row[val] for val in range(3, 7)]
        video = VideoFileStream(file_path, timestamps,
                                segment_len, output_shape)
        reader_pool.submit(video.get_all_segments, segments_q)
    # wait for all worker threads to complete
    reader_pool.shutdown()
    # indicate all done to consumer
    segments_q.put(None)
    logging.info('DONE: Finished reading all files!')


def write_segments(output_path, segments_q, segments_per_file=500,
                   thread_limit=2):
    '''Writes a video as batches to a single TFRecord
    Args:
    output_path: Path of output dir
    segments_q: queue.Queue to get segments from
    segments_per_file: #segments in a TFRecord
    thread_limit: max #threads for writing
    '''
    # Create dir if doesnt exist
    os.makedirs(output_path, exist_ok=True)
    # thread manager for writes
    writer_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_limit)
    # To keep track of #TFRecord written
    file_num = 0
    # Store all segments for a TFRecord
    segments = []
    while True:
        elem = segments_q.get()
        # Not needed now, might need if we make certain changes
        segments_q.task_done()
        if elem is not None:
            # Get the result fromt the Future
            segments.append(elem)
        if len(segments) == segments_per_file or (elem is None and len(segments) > 0):
            # write if limit or if last segment
            writer_pool.submit(write_tf_record, output_path,
                               file_num, segments)
            segments = []
            file_num += 1
        # Check if all segments are done
        if elem is None:
            break
    # wait for all write threads to complete
    writer_pool.shutdown()
    logging.info('DONE: Finished writing all segments!')


def write_tf_record(output_path, file_num, segments):
    '''Writes a video as batches to a single TFRecord
    Args:
    output_path: Path of output dir
    file_num: num for the TFRecord
    segments: a list of segments of a video
    '''
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
        segment = tf.convert_to_tensor(segments[itr]['segment'],
                                       dtype=tf.string)
        segment = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(segment).numpy()]
        ))
        name = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(segments[itr]['file'], 'utf-8')]
        ))
        num = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[segments[itr]['num']]
        ))
        label = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[segments[itr]['label']]
        ))
        feature = {
            'segment': segment,
            'file': name,
            'num': num,
            'label': label
        }
        record = tf.train.Example(features=tf.train.Features(feature=feature))
        # writing to TFRecord
        writer.write(record.SerializeToString())

    logging.debug(f'DONE:{rel_path}')
    writer.close()


def process_dir(data_frame, base_path, output_path, buffer_size=400,
                output_shape=(128, 128), segment_len=16,
                max_read_threads=32, max_write_threads=2, segments_per_file=500):
    '''Processes all the mp4 files in dir_path and outputs them as TFRecord files
    Args:
    data_frame: DataFrame containing all the videos to process along with timestamps
    base_path: path to the dir with the dirs for each class
    output_path: path of dir to output TFRecords to
    buffer_size: max #segments to buffer
    output_shape: dimensions of the processed frame
    segment_len: #frames in each segment
    max_read_threads: max #threads for reading
    max_write_threads: max #threads for writing
    segments_per_file: #segments in each TFRecord
    '''
    # To get and put segments to
    segments_q = queue.Queue(maxsize=buffer_size)
    read_kwargs = {'segment_len': segment_len,
                   'output_shape': output_shape,
                   'thread_limit': max_read_threads}
    read_thread = threading.Thread(target=read_segments,
                                   name='read_thread',
                                   args=(data_frame, base_path, segments_q),
                                   kwargs=read_kwargs)
    write_kwargs = {'segments_per_file': segments_per_file,
                    'thread_limit': max_read_threads}
    write_thread = threading.Thread(target=write_segments,
                                    name='write_thread',
                                    args=(output_path, segments_q),
                                    kwargs=write_kwargs)
    read_thread.start()
    write_thread.start()
    # Wait for threads to complete
    read_thread.join()
    write_thread.join()
    logging.info('All done!')


def get_data_frames(timestamp_file_path, num_validation):
    '''returns a list of dataframes, (valid, test)
    timestamp_file_path: path to the file with the timestamps and file names
    num_validation: #videos to use for the validation set
    '''
    df = pd.read_csv(timestamp_file_path, sep='  ',
                     names=['filename', 'classname', 'start1',
                            'end1', 'start2', 'end2'])
    split_df = train_test_split(df, train_size=num_validation,
                                stratify=df['classname'])
    return split_df


def make_valid_and_test(timestamp_file_path, base_path, output_paths,
                        num_validation=100, buffer_size=400, output_shape=(128, 128),
                        segment_len=16, max_read_threads=32,
                        max_write_threads=2, segments_per_file=500):
    '''Wraps 2 calls to process_dir to make validation and test set
    timestamp_file_path: path to the file with the timestamps and file names
    base_path: path to the dir with the dirs for each class
    output_paths: list of paths of dir to output TFRecords, [valid_path, test_path]
    num_validation: #videos to use for the validation set
    buffer_size: max #segments to buffer
    output_shape: dimensions of the processed frame
    segment_len: #frames in each segment
    max_read_threads: max #threads for reading
    max_write_threads: max #threads for writing
    segments_per_file: #segments in each TFRecord
    '''
    split_df = get_data_frames(timestamp_file_path, num_validation)
    # make validation set
    process_dir(split_df[0], base_path, output_paths[0], buffer_size,
                output_shape, segment_len, max_read_threads,
                max_write_threads, segments_per_file)
    # make test set
    process_dir(split_df[1], base_path, output_paths[1], buffer_size,
                output_shape, segment_len, max_read_threads,
                max_write_threads, segments_per_file)


if __name__ == "__main__":
    timestamp_file_path = './list.txt'
    base_path = './Testing'
    output_paths = ['./ValidSet', './TestSet']
    num_validation = 100
    buffer_size = 400
    output_shape = (128, 128)
    segment_len = 16
    max_read_threads = 8
    max_write_threads = 2
    segments_per_file = 500

    split_df = get_data_frames(timestamp_file_path, num_validation)
    # make validation set
    process_dir(split_df[0], base_path, output_paths[0], buffer_size,
                output_shape, segment_len, max_read_threads,
                max_write_threads, segments_per_file)
    # make test set
    process_dir(split_df[1], base_path, output_paths[1], buffer_size,
                output_shape, segment_len, max_read_threads,
                max_write_threads, segments_per_file)
