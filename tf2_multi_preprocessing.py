import os
import sys
import threading
import logging
import queue
import concurrent.futures
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
        self.segment_len = segment_len
        self.output_shape = output_shape
        self.index = index
        self.cap = cv2.VideoCapture(file_name)
        # Check is corrupt
        if not self.cap.isOpened():
            logging.fatal(f'{file_name} is CORRUPT. Delete it and retry.')
            raise Exception('Corrupt File')
        self.frame_count = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        # remainder frames will be dropped
        self.segments_left = int(self.frame_count) // self.segment_len
        # lock to prevent race conditions during multithreaded access
        self.lock = threading.Lock()
        logging.info(f'START:Reading:{self.file_name}\
                     :total_frames:{self.frame_count}')

    def dec_segments_left(self):
        '''Decrements the #segments_left by 1
        '''
        self.segments_left -= 1

    def finish(self, future):
        '''Clean up by closing the video stream
        '''
        self.cap.release()
        logging.info(f'DONE:Reading:{self.file_name}')

    def make_segment(self):
        '''Returns the next segment of self.segment_len frames
        '''
        # Wait till lock, prevents race conditions
        self.lock.acquire()
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
        # Free the lock for other threads
        self.lock.release()
        return segment


def read_segments(dir_path, segments_q, parallel_reads=950,
                  segment_len=16, output_shape=(128, 128),
                  thread_limit=32):
    '''Processes all the mp4 files in dir_path and outputs them as TFRecord files
    Args:
    dir_path: path of dir containing the input videos
    segments_q: queue.Queue to put segments into
    parallel_reads: #files to simultaneously read from
    segment_len: #frames in each segment
    output_shape: dimensions of the processed frame
    thread_limt: max #threads to use for reading
    '''
    # get paths to all mp4 s
    file_list = glob.glob(os.path.join(dir_path, '*.mp4'))
    # storing here as we do ops which modify the length
    len_file_list = len(file_list)
    logging.info(f'Processing {len_file_list} files in {dir_path}')
    videos = []
    for itr in range(min(len_file_list, parallel_reads)):
        videos.append(VideoFileStream(file_list.pop(),
                                      itr,
                                      segment_len=segment_len,
                                      output_shape=output_shape,
                                      ))
    # manages the threads for reads
    reader_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_limit)
    while len(videos) > 0:
        # select a random video
        rand_idx = random.randint(0, len(videos) - 1)
        if videos[rand_idx].segments_left == 0:
            logging.fatal(f'Tried to get segments when empty')
        videos[rand_idx].dec_segments_left()
        future = reader_pool.submit(videos[rand_idx].make_segment)
        if videos[rand_idx].segments_left == 0:
            # if last segment, do cleanup action
            future.add_done_callback(videos[rand_idx].finish)
            # swap to end and remove, for fast deletion
            videos[-1], videos[rand_idx] = videos[rand_idx], videos[-1]
            videos.pop()
            # replace with a new video, if available
            if len(file_list) > 0:
                videos.append(VideoFileStream(file_list.pop(),
                                              len(videos),
                                              segment_len=segment_len,
                                              output_shape=output_shape,
                                              ))
        segments_q.put(future)
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
            segments.append(elem.result())
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
        # Storing garbage for now
        name = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(file_name, 'utf-8')]
        ))
        # Storing garbage for now
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


def process_dir(dir_path, output_path, buffer_size=400,
                output_shape=(128, 128), segment_len=16, parallel_reads=900,
                max_read_threads=32, max_write_threads=2, segments_per_file=500):
    '''Processes all the mp4 files in dir_path and outputs them as TFRecord files
    Args:
    dir_path: path of dir containing the input videos
    out_path: path of dir to output TFRecords to
    buffer_size: max #segments to buffer
    output_shape: dimensions of the processed frame
    segment_len: #frames in each segment
    parallel_files: #files to simultaneously read from
    max_read_threads: max #threads for reading
    max_write_threads: max #threads for writing
    segments_per_file: #segments in each TFRecord
    '''
    # To get and put segments to
    segments_q = queue.Queue(maxsize=buffer_size)
    read_kwargs = {'parallel_reads': parallel_reads,
                   'segment_len': segment_len,
                   'output_shape': output_shape,
                   'thread_limit': max_read_threads}
    read_thread = threading.Thread(target=read_segments,
                                   name='read_thread',
                                   args=(dir_path, segments_q),
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


if __name__ == "__main__":
    dir_path = './Dataset_Samples'
    out_path = './tfrecords'
    process_dir(dir_path, out_path, max_read_threads=16)
