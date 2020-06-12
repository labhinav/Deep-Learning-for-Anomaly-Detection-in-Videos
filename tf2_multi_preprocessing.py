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
    def __init__(self, file_name, index, segment_len=16, output_shape=(128, 128)):
        self.file_name = file_name
        self.segment_len = segment_len
        self.output_shape = output_shape
        self.index = index
        self.cap = cv2.VideoCapture(file_name)
        self.frame_count = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        # remainder frames will be dropped
        self.segments_left = int(self.frame_count) // self.segment_len
        self.lock = threading.Lock()
        logging.info(f'START:Reading:{self.file_name}\
                     :total_frames:{self.frame_count}')

    def dec_segments_left(self):
        self.segments_left -= 1

    def finish(self, future):
        self.cap.release()
        logging.info(f'DONE:Reading:{self.file_name}')

    def make_segment(self):
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
        self.lock.release()
        return segment


def read_segments(dir_path, segments_q, parallel_reads=950,
                  segment_len=16, output_shape=(128, 128),
                  thread_limit=32):
    file_list = glob.glob(os.path.join(dir_path, '*.mp4'))
    len_file_list = len(file_list)
    logging.info(f'Processing {len_file_list} files in {dir_path}')
    videos = []
    for itr in range(min(len_file_list, parallel_reads)):
        videos.append(VideoFileStream(file_list.pop(),
                                      itr,
                                      segment_len=segment_len,
                                      output_shape=output_shape,
                                      ))
    reader_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_limit)
    while len(videos) > 0:
        rand_idx = random.randint(0, len(videos) - 1)
        if videos[rand_idx].segments_left == 0:
            logging.fatal(f'Tried to get segments when empty')
        videos[rand_idx].dec_segments_left()
        future = reader_pool.submit(videos[rand_idx].make_segment)
        if videos[rand_idx].segments_left == 0:
            future.add_done_callback(videos[rand_idx].finish)
            videos[-1], videos[rand_idx] = videos[rand_idx], videos[-1]
            videos.pop()
            if len(file_list) > 0:
                videos.append(VideoFileStream(file_list.pop(),
                                              len(videos),
                                              segment_len=segment_len,
                                              output_shape=output_shape,
                                              ))
        segments_q.put(future)
    reader_pool.shutdown()
    segments_q.put(None)
    logging.info('DONE: Finished reading all files!')


def write_segments(output_path, segments_q, segments_per_file=500,
                   thread_limit=2):
    writer_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_limit)
    file_num = 0
    segments = []
    while True:
        elem = segments_q.get()
        segments_q.task_done()
        if elem is not None:
            segments.append(elem.result())
        if len(segments) == segments_per_file or (elem is None and len(segments) > 0):
            writer_pool.submit(write_tf_record, output_path,
                               file_num, segments)
            segments = []
            file_num += 1
        if elem is None:
            break
    writer_pool.shutdown()
    logging.info('DONE: Finished writing all segments!')


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


def process_dir(dir_path, output_path, buffer_size=400,
                output_shape=(128, 128), segment_len=16, parallel_reads=900,
                max_read_threads=32, max_write_threads=2, segments_per_file=500):
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
    read_thread.join()
    write_thread.join()
    logging.info('All done!')


if __name__ == "__main__":
    dir_path = './Dataset_Samples'
    out_path = './tfrecords'
    process_dir(dir_path, out_path, max_read_threads=16)
