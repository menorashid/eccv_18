import sys
sys.path.append('./')
import os
from helpers import util, visualize
import glob
import numpy as np
import subprocess

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
	video_files = glob.glob(os.path.join(data_dir,'*.avi'))
	print video_files
	for video_file in video_files:
		save_frames(out_dir, video_file)


def main():
	pass;	

		

	
if __name__=='__main__':
	main()