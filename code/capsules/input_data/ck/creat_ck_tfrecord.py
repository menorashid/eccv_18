import util
import scipy.misc
import numpy as np
import tensorflow as tf
import os
import random

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(path, mean_im, std_im):
    image = scipy.misc.imread(path).astype(np.float32)
    image = image-mean_im
    image = image/std_im
    image = image[:,:,np.newaxis]
    return image

def main():
    data_dir = '../../../../data/ck_96/train_test_files'
    out_dir = '../../../../data/ck_96/train_test_files_tfrecords'
    im_path_prepend = '../../../'
    util.mkdir(out_dir)

    split_num = 0

    train_file = os.path.join(data_dir, 'train_'+str(split_num)+'.txt')
    test_file = os.path.join(data_dir,'test_'+str(split_num)+'.txt')
    mean_file = os.path.join(data_dir,'train_'+str(split_num)+'_mean.png')
    std_file =  os.path.join(data_dir,'train_'+str(split_num)+'_std.png')

    mean_im = scipy.misc.imread(mean_file).astype(np.float32)
    std_im = scipy.misc.imread(std_file).astype(np.float32)
    std_im[std_im==0]=1

    for in_file in [train_file,test_file]:
        
        out_file = os.path.join(out_dir,os.path.split(in_file)[1].replace('.txt','.tfrecords'))
        print in_file, out_file

        lines = util.readLinesFromFile(in_file)
        random.shuffle(lines)

        writer = tf.python_io.TFRecordWriter(out_file)
        for idx_line,line in enumerate(lines):
            if idx_line%100==0:
                print idx_line,line

            im_path, label = line.split(' ')
            label = int(label)
            im_path = os.path.join(im_path_prepend,im_path)

            img = load_image(im_path, mean_im, std_im)
            # print img.shape,np.min(img),np.max(img)

            feature = {'image_raw': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                    'label': _int64_feature(label),   
                       'height':  _int64_feature(img.shape[0]),
                      'width': _int64_feature(img.shape[1]),
                      'depth': _int64_feature(img.shape[2])
                       }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
    
        writer.close()



if __name__=='__main__':
    main()