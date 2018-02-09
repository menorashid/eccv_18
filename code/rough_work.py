import os
from helpers import util
import glob

def delete_some_files():
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


def main():
	out_file_sh = 'capsules/test_em_all.sh'
	all_commands = []
	for model_num in range(100,1100,100):
		# command_curr = 'python experiment.py --data_dir=../../data/ck_96/train_test_files_tfrecords/test_0.tfrecords --train=false --summary_dir=../../experiments/sabour_mnist/ck_attempt_0/results_%s --checkpoint=../../experiments/sabour_mnist/ck_attempt_0/train/model.ckpt-%s --hparams_override remake=0 --dataset ck' % (str(model_num),str(model_num))

		command_curr = 'python experiment.py --data_dir=../../data/ck_96/train_test_files_tfrecords/test_0.tfrecords --train=false --summary_dir=../../experiments/sabour_mnist/ck_attempt_0_baseline/results_%s --checkpoint=../../experiments/sabour_mnist/ck_attempt_0_baseline/train/model.ckpt-%s --hparams_override remake=0 --dataset ck --model=baseline --validate=true' % (str(model_num),str(model_num))
	
		all_commands.append(command_curr)

	util.writeFile(out_file_sh,all_commands)






if __name__=='__main__':
	main()