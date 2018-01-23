import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import subprocess
import scipy.io
import cv2
import matplotlib.pyplot as plt
import skimage.transform
import multiprocessing

def get_frame_au_anno(au_files):
	num_frames = len(util.readLinesFromFile(au_files[0]))
	anno = [[val] for val in range(num_frames)]
	for au_file in au_files:
		au_curr = os.path.split(au_file)[1]
		au_curr = int(au_curr[:-4].split('_')[-1][2:])
		
		lines = util.readLinesFromFile(au_file)
		for line in lines:
			frame, intensity = line.split(',')

			intensity = int(intensity)
			if intensity>0:
				anno[int(frame)-1].append(au_curr)
				anno[int(frame)-1].append(intensity)

	anno = [anno_curr for anno_curr in anno if len(anno_curr)>1]
	return anno

def get_emotion_count():
	anno_video_all = []
	meta_anno_dir = '../data/disfa/ActionUnit_Labels'
	video_anno_dirs = glob.glob(os.path.join(meta_anno_dir,'*'))
	for video_anno_dir in video_anno_dirs:
		au_files = glob.glob(os.path.join(video_anno_dir,'*.txt'))
		au_files.sort()
		anno_video = get_frame_au_anno(au_files)
		anno_video_all.extend(anno_video)

	# print len(anno_video_all)
	# return

	emotion_lists = [[6,12],[1,4,15],[1,2,5,26],[1,2,4,5,20,26],[9,15]]
	emotion_lists = emotion_lists+ [[1,15],[4,15],[1,2,5],[1,2,26],[1,2,4,20],[1,2,4,5,20],[1,2,5,20]]
	# for emotion_list in emotion_lists:
	emotion_lists = [[6,12],
					[4,5,7,22,23,24],
					[4,5,22,23,24],
					[4,7,22,23,24],
					[1,4,15],
					[1,4,15,17],
					[9,25,26],
					[10,25,26],
					[9,10,25,26],
					[1,2,4,5,7,20,25,26],
					[1,2,4,5,7,20,26],
					[1,2,5,25,26],
					[1,2,5,26]]

	
	anno_combination_list = []

	for anno_curr in anno_video_all:
		aus = anno_curr[1::2]
		aus.sort()
		aus = [str(au_curr) for au_curr in aus]
		anno_combination_list.append(' '.join(aus))
	
	emotion_lists = [' '.join([str(val) for val in emotion_list]) for emotion_list in emotion_lists]
	for anno_combo in emotion_lists:
	# set(anno_combination_list):
		print anno_combo,anno_combination_list.count(anno_combo)

def save_frames(out_dir,video_file,out_size=None):
	video_name = os.path.basename(video_file)
	video_name = video_name[:video_name.rindex('.')]

	out_dir = os.path.join(out_dir,video_name)
	util.mkdir(out_dir)
	frame_name = os.path.join(out_dir,video_name+'_%05d.jpg')

	command = []
	command.extend(['ffmpeg','-i'])
	command.append(video_file)
	command.append(frame_name)
	if out_size is not None:
		command.extend(['-s',str(out_size[0])+'x'+str(out_size[1])])
	command.append('-hide_banner')
	command = ' '.join(command)
	print command
	subprocess.call(command, shell=True)

def script_save_frames():
	data_dir = '../data/disfa/Videos_LeftCamera'
	out_dir = data_dir+'_frames'
	util.mkdir(out_dir)
	# video_files = glob.glob(os.path.join(data_dir,'*.avi'))
	# print video_files
	# for video_file in video_files:
	video_file = '../data/disfa/Videos_LeftCamera/LeftVideoSN013_comp.avi'
	save_frames(out_dir, video_file)


def script_view_bad_kp():
	out_dir = '../scratch/disfa/kp_check'
	util.makedirs(out_dir)
	out_file = os.path.join(out_dir,'SN001_0000_lm.jpg')
	mat_dir = '../data/disfa/Landmark_Points' 
	# SN001 'frame_lm'
	frame_dir ='../data/disfa/Videos_LeftCamera_frames'
	
	problem_dict = {'SN030':[ 939, 962, 1406, 1422, 2100, 2132, 2893, 2955],
					'SN029':[ 4090, 4543],
					'SN028':[ 1875, 1885, 4571, 4690],
					'SN027':[ 3461, 3494, 4738, 4785],
					'SN025':[ 4596, 4662, 4816, 4835],
					'SN023':[ 1021, 1049, 3378, 3557, 3584, 3668, 4547, 4621, 4741, 4772, 4825, 4845],
					'SN021':[ 574, 616, 985, 1164, 1190, 1205, 1305, 1338, 1665, 1710, 1862, 2477, 2554, 4657, 4710, 4722],
					'SN011':[ 4529, 4533, 4830, 4845,  ],
					'SN009':[ 1736, 1808, 1851, 1885],
					'SN006':[ 1349, 1405],
					'SN004':[ 4541, 4555],
					'SN002':[ 800, 826],
					'SN001':[ 398, 420, 3190, 3243]}

	for video_name in problem_dict.keys():
		range_starts = problem_dict[video_name][::2]
		range_ends = problem_dict[video_name][1::2]
				
		out_dir_curr = os.path.join(out_dir,video_name)
		util.mkdir(out_dir_curr)
		print video_name
		for idx_range in range(len(range_starts)):
			print range_starts[idx_range],range_ends[idx_range]

			for anno_num in range(range_starts[idx_range]-1,range_ends[idx_range]):


	# for anno_num in range(397,421)+range(3189,3244):
				str_num_mat = '0'*(4-len(str(anno_num)))+str(anno_num)
				str_num_im = '0'*(5-len(str(anno_num+1)))+str(anno_num+1)
				
				
				mat_file = os.path.join(mat_dir,video_name,'tmp_frame_lm',video_name+'_'+str_num_mat+'_lm.mat')
				if not os.path.exists(mat_file):
					mat_file = os.path.join(mat_dir,video_name,'tmp_frame_lm','l0'+str_num_mat+'_lm.mat')

				im_file = os.path.join(frame_dir,'LeftVideo'+video_name+'_comp','LeftVideo'+video_name+'_comp_'+str_num_im+'.jpg')

				out_file = os.path.join(out_dir_curr,video_name+'_'+str_num_mat+'_provided.jpg')
				if os.path.exists(out_file):
					continue
				
				im = scipy.misc.imread(im_file)
				pts = scipy.io.loadmat(mat_file)
				pts = pts['pts']
				for pt_curr in pts:
					pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
					cv2.circle(im, pt_curr, 5, (255,0,0),-1)


				scipy.misc.imsave(out_file,im)	

		visualize.writeHTMLForFolder(out_dir_curr)

def script_save_avg_kp():
	out_dir = '../data/disfa'
	all_kp_files = glob.glob(os.path.join('../data/disfa/Landmark_Points','*','*','*.mat'))
	all_kp = None
	# all_kp_files = all_kp_files[::100]
	for kp_file in all_kp_files:
		kp = scipy.io.loadmat(kp_file)['pts']
		kp = kp-np.min(kp,0)
		if all_kp is None:
			all_kp = kp
		else:
			all_kp = all_kp+kp

	avg_kp = all_kp/len(all_kp_files)
	
	out_file = os.path.join(out_dir,'avg_kp.npy')
	np.save(out_file,avg_kp)

	plt.figure()
	plt.plot(avg_kp[:,0],np.max(avg_kp[:,1])-avg_kp[:,1],'*b')
	plt.savefig(os.path.join(out_dir,'avg_kp.jpg'))
	plt.close()

def script_save_avg_kp():
	out_dir = '../data/disfa'
	avg_kp = np.load(os.path.join(out_dir,'avg_kp.npy'))
	avg_kp = avg_kp - np.min(avg_kp,0)
	avg_kp = avg_kp /np.max(avg_kp,0)
	ratios = [75,20,5]
	assert sum(ratios)==100
	center = ratios[0]*200/100
	top = ratios[1]*200/100
	side = (100-ratios[0])/2. * 200/100
	avg_kp = avg_kp*center
	avg_kp[:,1]= avg_kp[:,1]+top
	avg_kp[:,0]= avg_kp[:,0]+side
	print side
	out_name = 'avg_kp_200_'+'_'.join([str(val) for val in ratios])
	out_file = os.path.join(out_dir,out_name+'.npy')
	np.save(out_file,avg_kp)
	
	plt.figure()
	plt.plot(avg_kp[:,0],avg_kp[:,1],'*b')
	plt.savefig(os.path.join('../scratch/disfa/kp_check', out_name+'.jpg'))
	plt.close()


def save_registered_face((avg_pts_file,mat_file,im_file,out_file,idx)):
	if not idx%100:
		print idx

	avg_pts = np.load(avg_pts_file)

	pts = scipy.io.loadmat(mat_file)
	pts = pts['pts']

	im = scipy.misc.imread(im_file)
	
	tform = skimage.transform.estimate_transform('similarity', pts, avg_pts)
	im_new = skimage.transform.warp(im, tform.inverse, output_shape=(200,200), order=1, mode='edge')

	# # print im.shape
	# im_new = im_new*255
	# for pt_curr in avg_pts:
	# 	pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
	# 	print pt_curr
	# 	cv2.circle(im_new, pt_curr, 3, (255,0,0),-1)

	# out_file = '../scratch/disfa/warp.jpg'
	scipy.misc.imsave(out_file,im_new)


def script_save_registered_faces():
	mat_dir_meta = '../data/disfa/Landmark_Points' 
	frame_dir_meta ='../data/disfa/Videos_LeftCamera_frames'
	out_dir_meta = '../data/disfa/Videos_LeftCamera_frames_200'
	util.mkdir(out_dir_meta)

	avg_pts_file = '../data/disfa/avg_kp_200_75_20_5.npy'
	video_names = [dir_curr for dir_curr in os.listdir(mat_dir_meta) if os.path.isdir(os.path.join(mat_dir_meta,dir_curr))]

	args = []
	
	for video_name in video_names:
		mat_dir = os.path.join(mat_dir_meta,video_name,'tmp_frame_lm')
		mat_files = glob.glob(os.path.join(mat_dir,'*.mat'))
		im_files = glob.glob(os.path.join(frame_dir_meta,'LeftVideo'+video_name+'_comp','*.jpg'))
		im_files.sort()
		mat_files.sort()
		if len(mat_files)>len(im_files):
			mat_files = mat_files[1:]
		assert len(mat_files)==len(im_files)
		

		for idx_mat_file,(mat_file,im_file) in enumerate(zip(mat_files,im_files)):
			mat_num = int(mat_file[-11:-7])
			im_num = int(im_file[-8:-4])
			if im_num!=mat_num:
				assert im_num-mat_num==1

			out_file = im_file.replace(frame_dir_meta,out_dir_meta)
			if not os.path.exists(out_file):
				util.makedirs(os.path.dirname(out_file))
				args.append((avg_pts_file,mat_file,im_file,out_file,len(args)))
	
	print len(args)

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(save_registered_face,args)

def main():
	# script_save_frames()

	script_save_registered_faces()
	# script_save_avg_kp()
	return
	out_dir = '../data/disfa'
	# avg_pts = np.load(os.path.join(out_dir,'avg_kp_200_80.npy'))
	avg_pts = np.load(os.path.join(out_dir,'avg_kp_200_75_20_5.npy'))
	mat_file = '../data/disfa/Landmark_Points/SN001/tmp_frame_lm/SN001_0000_lm.mat'

	pts = scipy.io.loadmat(mat_file)
	pts = pts['pts']
	# print np.min(pts,0),np.max(pts,0)
	

	# # im_file = '../data/disfa/Video_RightCamera_frames/RightVideoSN001_Comp/RightVideoSN001_Comp_00001.jpg'
	im_file = '../data/disfa/Videos_LeftCamera_frames/LeftVideoSN001_comp/LeftVideoSN001_comp_00001.jpg'
	im = scipy.misc.imread(im_file)
	import skimage.transform
	tform = skimage.transform.estimate_transform('similarity', pts, avg_pts)

	im_new = skimage.transform.warp(im, tform.inverse, output_shape=(200,200), order=1, mode='edge')

	# print im.shape
	im_new = im_new*255
	for pt_curr in avg_pts:
		pt_curr = (int(pt_curr[0]),int(pt_curr[1]))
		print pt_curr
		cv2.circle(im_new, pt_curr, 3, (255,0,0),-1)

	out_file = '../scratch/disfa/warp.jpg'
	scipy.misc.imsave(out_file,im_new)









		

	
if __name__=='__main__':
	main()