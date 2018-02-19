import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import scipy.misc

def write_facs_file(in_file,out_file,facs_dir):
	im_files= util.readLinesFromFile(in_file)

	out_anno = []

	for im_file in im_files:
		im_file = im_file.split(' ')[0]
		im_name_split = os.path.split(im_file)[1][:-4].split('_')
		facs_file = os.path.join(facs_dir,im_name_split[0],im_name_split[1],'_'.join(im_name_split)+'_facs.txt')

		if not os.path.exists(facs_file):
			continue

		facs_anno = util.readLinesFromFile(facs_file)
		anno = []
		for line_curr in facs_anno:
			line_curr = line_curr.strip().split()
			anno = anno+[str(int(float(val))) for val in line_curr]
		anno_curr = ' '.join([im_file]+anno)
		out_anno.append(anno_curr)

		
	util.writeFile(out_file,out_anno)

def save_mean_std_vals:
	for split_num in range(0,1):
        train_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'.txt'
        test_file = '../data/ck_96/train_test_files/test_'+str(split_num)+'.txt'
        mean_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean.png'
        std_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_std.png'
        out_file = '../data/ck_96/train_test_files/train_'+str(split_num)+'_mean_std_val_0_1.npy'

        lines = util.readLinesFromFile(train_file)
        im_all = []
        for line in lines:
            im = line.split(' ')[0]
            im = scipy.misc.imread(im).astype(np.float32)
            im = im/255.
            im = im[:,:,np.newaxis]
            im_all.append(im)

        print len(im_all)
        im_all = np.concatenate(im_all,2)
        print im_all.shape, np.min(im_all),np.max(im_all)
        mean_val = np.mean(im_all)
        std_val = np.std(im_all)
        print mean_val,std_val
        mean_std = np.array([mean_val,std_val])
        print mean_std.shape, mean_std
        np.save(out_file,mean_std)

def main():
	data_dir = '../data/ck_96/train_test_files'
	train_file = os.path.join(data_dir,'train_0.txt')
	
	train_data = util.readLinesFromFile(train_file)
	train_data = [int(line_curr.split(' ')[1]) for line_curr in train_data]
	print set(train_data)

	return
	facs_anno_dir = '../data/ck_original/FACS'



	all_files = []
	fold_num = 0
	for fold_num in range(10):
		for file_pre in ['train','test']:
			in_file = os.path.join(data_dir,file_pre+'_'+str(fold_num)+'.txt')
			out_file = os.path.join(data_dir,file_pre+'_facs_'+str(fold_num)+'.txt')
			write_facs_file(in_file,out_file,facs_anno_dir)
			print in_file,out_file



if __name__=='__main__':
	main()