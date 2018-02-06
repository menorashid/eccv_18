import os
from helpers import util
import glob

def main():
	print 'hello'
	problem_dir = '/home/maheenrashid/eccv_18/experiments/sabour_mnist/attempt0_no_remake/train'

	ckpts = glob.glob(os.path.join(problem_dir,'model.ckpt-*.meta'))
	ckpts_int = [int(filename[filename.index('-')+1:filename.rindex('.')]) for filename in ckpts]

	# ckpts_int.sort()
	# print len(ckpts_int),ckpts_int[0],ckpts_int[-1]
	# print len(ckpts),ckpts[0]

	ckpts_int = [val for val in ckpts_int if val<=50000]

	str_keep = []
	for ckpt in ckpts:
		model_num = int(ckpt[ckpt.index('-')+1:ckpt.rindex('.')])
		if model_num<=50000 or model_num==150000 or model_num==100000:
			str_keep.append(ckpt[:ckpt.rindex('.')])

	print len(str_keep)

	all_files = glob.glob(os.path.join(problem_dir,'model.ckpt-*'))
	print 'FILES Before',len(all_files)
	for file_curr in all_files:
		start_str = file_curr[:file_curr.rindex('.')]
		if start_str not in str_keep:
			os.remove(file_curr)

	all_files = glob.glob(os.path.join(problem_dir,'model.ckpt-*'))
	print 'FILES After',len(all_files)
	




if __name__=='__main__':
	main()