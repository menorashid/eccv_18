import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob

def main():
	data_dir = '../data/ck_96/train_test_files'
	facs_anno_dir = '../data/ck_original/FACS'

	all_files = []
	fold_num = 0
	train_file = os.path.join(data_dir,'train_'+str(fold_num)+'.txt')
	test_file = os.path.join(data_dir,'test_'+str(fold_num)+'.txt')
	all_files = all_files+util.readLinesFromFile(train_file)+util.readLinesFromFile(test_file)
	assert len(all_files)==len(set(all_files))

	existence = []
	for file_curr in all_files:
		file_curr_split = file_curr.split(' ')
		anno = file_curr_split[1]
		im_name = os.path.split(file_curr_split[0])[1]
		im_name_split = im_name[:im_name.rindex('.')].split('_')
		facs_file = os.path.join(facs_anno_dir,im_name_split[0],im_name_split[1],'_'.join(im_name_split)+'_facs.txt')
		print facs_file, os.path.exists(facs_file),anno
		existence.append(os.path.exists(facs_file))

	print len(existence)
	print sum(existence)

		

if __name__=='__main__':
	main()