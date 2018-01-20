import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob

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

			

def main():
	data_dir = '../data/ck_96/train_test_files'
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