import json
import random
import os
import sys
import argparse

import time


def split_seen_unseen_verbs(test_split, seen_verbs):
	test_seen_split = {}
	test_unseen_split = {}
	for vid_seg_id in test_split.keys():
		if test_split[vid_seg_id]["verb"] in seen_verbs:
			test_seen_split[vid_seg_id] = test_split[vid_seg_id]
		else:
			test_unseen_split[vid_seg_id] = test_split[vid_seg_id]
	return test_seen_split, test_unseen_split


def split_seen_unseen_vids(test_split, seen_vids):
	test_seen_split = {}
	test_unseen_split = {}
	for vid_seg_id in test_split.keys():
		if test_split[vid_seg_id]["vid"] in seen_vids:
			test_seen_split[vid_seg_id] = test_split[vid_seg_id]
		else:
			test_unseen_split[vid_seg_id] = test_split[vid_seg_id]
	return test_seen_split, test_unseen_split


def calculate_verbs_rank(frame_verb_stats, Frame=None):
	verbs_video_clips = {}
	if Frame is not None:
		F = [Frame]
	else:
		F = list(frame_verb_stats.keys())

	for f in F:
		for v, D in frame_verb_stats[f]["verbs"].items():
			verbs_video_clips[v] = 0
			for d in D.keys():
				d_vid_counts = sum(list(frame_verb_stats[f]["verbs"][v][d].values()))
				verbs_video_clips[v] += d_vid_counts

	verbs_rank, instances = zip(*sorted(verbs_video_clips.items(), key=lambda item: item[1], reverse=True))
	return verbs_rank, instances


def split_rank(test_split, frame_verb_stats, top_low=[10, -10], overall=False):
	test_split_top = {}
	test_split_low = {}

	# select top, low verb classes
	if overall:
		# TODO frame disambiguation
		verbs_rank, instances = calculate_verbs_rank(frame_verb_stats)
		verbs_rank_top = verbs_rank[:top_low[0]]
		verbs_rank_low = verbs_rank[top_low[-1]:]
		
		for vid_seg_id in test_split.keys():
			if test_split[vid_seg_id]["verb"] in verbs_rank_top:
				test_split_top[vid_seg_id] = test_split[vid_seg_id]
			elif test_split[vid_seg_id]["verb"] in verbs_rank_low:
				test_split_low[vid_seg_id] = test_split[vid_seg_id]

	# # select top, low verb classes within a FN frame
	# else:
	# 	for F, v in test_split.items():
	# 		verbs_rank, instances = calculate_verbs_rank(frame_verb_stats, Frame=F)
	# 		verbs = v.keys()  # more than one verb for this FN frame
	# 		if len(verbs) > 1:
	# 			verbs_rank_top = verbs_rank[:top_low[0]]
	# 			verbs_rank_low = verbs_rank[top_low[-1]:]
	#
	# 			for verb, vid_clips in v.items():
	# 				if verb in verbs_rank_top:
	# 					if F not in test_split_top:
	# 						test_split_top[F] = {}
	# 					if verb not in test_split_top[F]:
	# 						test_split_top[F][verb] = vid_clips
	#
	# 				elif verb in verbs_rank_low:
	# 					if F not in test_split_low:
	# 						test_split_low[F] = {}
	# 					if verb not in test_split_top[F]:
	# 						test_split_low[F][verb] = vid_clips
	return test_split_top, test_split_low


def split_by_vid_seg(vid_segs_dict, ratio=None):
	# decide split based on vid seg id (vid + "_" + vid_seg)
	vid_seg_ids = list(vid_segs_dict.keys())
	random.shuffle(vid_seg_ids)
	total_num = len(vid_seg_ids)
	
	train_split_num = int(len(vid_seg_ids) * ratio["train"])

	train_vid_segs = vid_seg_ids[:train_split_num]
	train = {vid_seg_id: vid_segs_dict[vid_seg_id] for vid_seg_id in train_vid_segs}
	
	val_test_vid_segs = vid_seg_ids[train_split_num:]
	val_test = {vid_seg_id: vid_segs_dict[vid_seg_id] for vid_seg_id in val_test_vid_segs}
	
	# re-structure into domain-based such that the domain ratios are similar between val and test set
	vid_clips_by_domains = re_structure(val_test)
	test_vid_segs = []
	val_vid_segs = []
	test_val_ratio = int(round(ratio["test"] / ratio["val"], 0))
	div = test_val_ratio + 1
	for d in vid_clips_by_domains.keys():
		domain_vid_seg_ids = list(vid_clips_by_domains[d].keys())
		random.shuffle(domain_vid_seg_ids)
		# split into half
		test_split_num = int(len(domain_vid_seg_ids)/div)*test_val_ratio
		d_test_vid_segs = domain_vid_seg_ids[:test_split_num]
		d_val_vid_segs = domain_vid_seg_ids[test_split_num:]
		test_vid_segs += d_test_vid_segs
		val_vid_segs += d_val_vid_segs
	
	val = {vid_seg_id: vid_segs_dict[vid_seg_id] for vid_seg_id in val_vid_segs}
	test = {vid_seg_id: vid_segs_dict[vid_seg_id] for vid_seg_id in test_vid_segs}
	
	assert set(train_vid_segs).isdisjoint(test_vid_segs)
	assert total_num == len(test_vid_segs) + len(val_vid_segs) + len(train_vid_segs)
	
	return test, val, train


def re_structure(subset):
	vid_clips_by_domains = {}
	for vid_seg_id, vid_clip_info in subset.items():
		d = vid_clip_info["domain"]
		if d not in vid_clips_by_domains:
			vid_clips_by_domains[d] = {}
		if vid_seg_id not in vid_clips_by_domains[d]:
			vid_clips_by_domains[d][vid_seg_id] = vid_clip_info
	return vid_clips_by_domains


def get_verb_vid_segs(domain_dict):
	verb_vid_segs_dict = {}
	for d in domain_dict.keys():
		for vid in domain_dict[d].keys():
			for vid_clip in domain_dict[d][vid]:
				vid_seg_id = vid_clip["vid"] + "_" + str(vid_clip["vid seg"])
				if vid_seg_id not in verb_vid_segs_dict:
					verb_vid_segs_dict[vid_seg_id] = vid_clip
	return verb_vid_segs_dict


def get_domain_vid_segs(vid_seg_dict):
	domain_dict = {}
	for vid_seg_id in vid_seg_dict.keys():
		d = vid_seg_dict[vid_seg_id]["domain"]
		if d not in domain_dict:
			domain_dict[d] = set()
		domain_dict[d].add(vid_seg_id)
		
	for d in domain_dict.keys():
		domain_dict[d] = list(domain_dict[d])
	return domain_dict


def select_vids(frames_video_clips, verbs_vid_seg_info, F, verbs, train_split, val_split, test_split, seen=True):
	
	for verb in verbs:
		if verb in verbs_vid_seg_info:
			continue
		else:
			verbs_vid_seg_info[verb] = {"train": {}, "val": {}, "test": {}}
			
		verb_vid_segs_dict = get_verb_vid_segs(frames_video_clips[F][verb])
		
		print(f'splitting instances for {verb} (seen: {seen})')
		verb_vid_segs_num = len(verb_vid_segs_dict.keys())
		print(f'{verb_vid_segs_num} unique video segments for {verb} across domains')
		
		if seen:
			split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
		else:
			split_ratio = {"train": 0.0, "val": 0.5, "test": 0.5}
			
		test, val, train = split_by_vid_seg(verb_vid_segs_dict, ratio=split_ratio)
		
		assert verb_vid_segs_num == len(train.keys()) + len(val.keys()) + len(test.keys()), \
			f'verb: {verb}, all: {verb_vid_segs_num} train: {len(train.keys())}, val:{len(val.keys())}, test: {len(test.keys())}'

		# accumulate instances into train, val, test set respectively (vid_seg_id based)
		train_split.update(train)
		val_split.update(val)
		test_split.update(test)
		
		# keep track of verb train/val/test table for evaluation purpose
		verbs_vid_seg_info[verb]["train"].update(get_domain_vid_segs(train))
		verbs_vid_seg_info[verb]["val"].update(get_domain_vid_segs(val))
		verbs_vid_seg_info[verb]["test"].update(get_domain_vid_segs(test))
		
		# verbs_vid_seg_info[verb]["train"] += list(train.keys())
		# verbs_vid_seg_info[verb]["val"] += list(val.keys())
		# verbs_vid_seg_info[verb]["test"] += list(test.keys())
		
	return train_split, val_split, test_split, verbs_vid_seg_info


def calculate_ratios(domain_vids, domains):
	all = sum([num for num in domain_vids.values()])
	denom = all / len(domains)
	ratio = [(d, round(domain_vids[d] / denom, 2)) for d in domains]
	return ratio


def criteria_check(train, val, test, domains):
	train_vids = set()
	val_vids = set()
	test_vids = set()
	
	train_vid_segs = set()
	val_vid_segs = set()
	test_vid_segs = set()

	train_nouns = set()
	val_nouns = set()
	test_nouns = set()

	train_domains = {d: 0 for d in domains}
	val_domains = {d: 0 for d in domains}
	test_domains = {d: 0 for d in domains}

	for split, vids, vid_segs, nouns, split_domains in zip([train, val, test],
	                                                       [train_vids, val_vids, test_vids],
	                                                       [train_vid_segs, val_vid_segs, test_vid_segs],
	                                                       [train_nouns, val_nouns, test_nouns],
	                                                       [train_domains, val_domains, test_domains]):
		for vid_seg_id in split.keys():
			vid_segs.add(vid_seg_id)
			# check unseen vids
			vids.add(split[vid_seg_id]["vid"])
			# check unseen objects
			for n in split[vid_seg_id]["nouns"]:
				nouns.add(n)
			# check domain distribution
			split_domains[split[vid_seg_id]["domain"]] += 1
		

	# all_vids = train_vids | val_vids | test_vids
	# all_nouns = train_nouns | val_nouns | test_nouns

	print(f'val set: {round(len(val_vids - train_vids) / len(val_vids), 2) * 100}% of unseen train videos, '
				f'test set: {round(len(test_vids - train_vids) / len(test_vids), 2) * 100}% of unseen train videos, '
				f'{round(len(test_vids - val_vids) / len(test_vids), 2) * 100}% of unseen val videos.')

	print(f'val set: {round(len(val_vid_segs - train_vid_segs) / len(val_vid_segs), 2) * 100}'
	            f'% of unseen train video segments, '
				f'test set: {round(len(test_vid_segs - train_vid_segs) / len(test_vid_segs), 2) * 100}'
				f'% of unseen train video segments, '
				f'{round(len(test_vid_segs - val_vid_segs) / len(test_vid_segs), 2) * 100}% of unseen val video segments.')

	print(f'val set: {round(len(val_nouns - train_nouns) / len(val_nouns), 2) * 100}% of unseen train objects, '
				f'test set: {round(len(test_nouns - train_nouns) / len(test_nouns), 2) * 100}% of unseen train objects, '
				f'{round(len(test_nouns - val_nouns) / len(test_nouns), 2) * 100}% of unseen val objects.')

	print(f'train set domain ratios: {calculate_ratios(train_domains, domains)}')
	print(f'val set domain ratios: {calculate_ratios(val_domains, domains)}')
	print(f'test set domain ratios: {calculate_ratios(test_domains, domains)}')
	
	return train_vids, val_vids, test_vids


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--video_clips", type=str, default="single_result_verbs_video_clips.json", required=True,
						help="result verb video clips")
	parser.add_argument("--frame_verb_stats", type=str, default="single_frames_verbs_stats.json", required=True,
						help="statistics of verbs distribution across frames and video domains")
	parser.add_argument("--fixed_seen_verb_list", type=str, default="meta_info/kinectics400_joint_verb_labels.txt"
	                    , required=False, help="fix the seen verb classes by offer a list of result verbs"
	                                           ", note that the seen verbs v.s. unseen verbs ratio "
	                                           "across Frames won't be the same.")
	# parser.add_argument("--verb_noun_stats", type=str, default="single_verbs_nouns_stats.json", required=True,
	# 					help="statistics of verbs nouns distribution across frames and video domains")
	parser.add_argument("--eval_subsets", type=str, default="True",
						help="categorize test set into different evaluation subsets")
	parser.add_argument("--seeds", type=str, default="42,83,928", required=True,
						help="randomly split train-val-test set under multiple seeds")
	parser.add_argument('--categories', type=str, default="family,food,hobbies, ...", required=True,
						help="video domains used here")
	parser.add_argument("--output_dir", type=str, default="dataset/exp/", required=True,
						help="dataset split directory")

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	F_verbs_video_clips = json.load(open(args.video_clips, 'r'))
	frame_verb_stats = json.load(open(args.frame_verb_stats, 'r'))
	fixed_seen_verb_list = args.fixed_seen_verb_list
	
	if fixed_seen_verb_list is not None:
		with open(fixed_seen_verb_list, "r") as f:
			fixed_seen_verbs = f.readlines()
			fixed_seen_verbs = set([v.strip("\n") for v in fixed_seen_verbs])

	# verb_noun_stats = args.verb_noun_stats
	eval_subsets = args.eval_subsets
	seeds = args.seeds.split(",")
	output_dir = args.output_dir
	domains = args.categories.split(",")

	for seed_num in seeds:
		print(f'perform data splitting with seed {seed_num}.')
		os.makedirs(os.path.join(output_dir, seed_num), exist_ok=True)

		random.seed(int(seed_num))

		train_split = dict()
		val_split = dict()
		test_split = dict()

		seen_verbs = set()
		unseen_verbs = set()
		all_verbs = set()

		# for evaluation purpose
		eval_tabel = dict()
		verbs_vid_seg_info = dict()
		
		print(f'start splitting')
		start = time.time()
		for F in frame_verb_stats.keys():
			print(F)
			F_verbs = set(frame_verb_stats[F]["verbs"].keys())
			all_verbs.update(F_verbs)

			# see the verb classes' ranks within a Frame => for splitting seen & unseen verb classes
			# verbs_rank, instances = calculate_verbs_rank(frame_verb_stats, Frame=F)
			
			# alternative: assign 1/2 of verb classes in different rank bins to train set?
			# assign 80% of verb classes to train set
			seen_class_ratio = 0.8
			seen_class_num = int(round(len(F_verbs) * seen_class_ratio, 0))
			unseen_class_num = len(F_verbs) - seen_class_num
			
			# print("FN num", len(F_verbs))
			# print("FN seen num", seen_class_num)

			# get seen verb classes
			FN_seen_verbs = set()
			FN_unseen_verbs = set()
			
			# already in seen verb classes provided by external seen verbs list
			FN_seen_verbs.update(F_verbs.intersection(fixed_seen_verbs))

			# already in overall seen/unseen verb classes given that one verb share multiple FN frames
			FN_seen_verbs.update(F_verbs.intersection(seen_verbs))
			FN_unseen_verbs.update(F_verbs.intersection(unseen_verbs))
			# print("FN seen", len(FN_seen_verbs))
			# print("FN unseen", len(FN_unseen_verbs))
			
			seen_to_choose_num = seen_class_num - len(FN_seen_verbs)
			to_choose = list(F_verbs - FN_seen_verbs - FN_unseen_verbs)
			# print("to choose", len(to_choose))
			
			while len(FN_seen_verbs) < seen_to_choose_num and len(to_choose) > 0:
				verb = random.choice(to_choose)
				FN_seen_verbs.add(verb)
				to_choose = [v for v in to_choose if v != verb]
				# print("to choose", len(to_choose))
				# print("FN seen +num", len(FN_seen_verbs))
			
			FN_unseen_verbs.update(set(to_choose))
			# print("FN unseen +num", len(FN_unseen_verbs))
			
			# keep track of overall seen verbs & unseen verbs
			seen_verbs.update(FN_seen_verbs)
			unseen_verbs.update(FN_unseen_verbs)
			
			assert len(F_verbs) == len(FN_seen_verbs) + len(FN_unseen_verbs)
			assert FN_seen_verbs.isdisjoint(FN_unseen_verbs)
			
			# unseen verb class: split according to video segment ids,
			# the instance ratio of train-val-test split follow 0% -50% -50% split
			train_split, val_split, test_split, verbs_vid_seg_info = \
				select_vids(F_verbs_video_clips, verbs_vid_seg_info, F,
				            FN_unseen_verbs, train_split, val_split, test_split, seen=False)
			
			# seen verb class: split according to video segment ids,
			# the instance ratio of train-val-test split follow 80% -10% -10% split.
			train_split, val_split, test_split, verbs_vid_seg_info = \
				select_vids(F_verbs_video_clips, verbs_vid_seg_info, F,
				            FN_seen_verbs, train_split, val_split, test_split, seen=True)

			eval_tabel[F] = {"seen classes": {}, "unseen classes": {}}
			for v in FN_seen_verbs:
				eval_tabel[F]["seen classes"][v] = verbs_vid_seg_info[v]
			for v in FN_unseen_verbs:
				eval_tabel[F]["unseen classes"][v] = verbs_vid_seg_info[v]
				
		end = time.time()
		print(f'end splitting, which took {end-start} secs in total')
		
		assert len(all_verbs) == len(seen_verbs) + len(unseen_verbs)
		assert seen_verbs.isdisjoint(unseen_verbs)

		print(f'seen verbs counts: {len(seen_verbs)}')
		print(f'unseen verbs counts: {len(unseen_verbs)}')

		train_vids, val_vids, test_vids = criteria_check(train_split, val_split, test_split, domains)

		train_instances = len(train_split.keys())
		val_instances = len(val_split.keys())
		test_instances = len(test_split.keys())
		
		print(f'train video clip instances:, {train_instances}')
		print(f'val video clip instances:, {val_instances}')
		print(f'test video clip instances:, {test_instances}')

		train_file_path = os.path.join(output_dir, str(seed_num), 'train', 'train.json')
		json.dump(train_split, open(train_file_path, 'w'))
		print(f'save train into {train_file_path}')

		val_file_path = os.path.join(output_dir, str(seed_num), 'val', 'val.json')
		json.dump(val_split, open(val_file_path, 'w'))
		print(f'save val set into {val_file_path}')

		test_file_path = os.path.join(output_dir, str(seed_num), 'test', 'test.json')
		json.dump(test_split, open(test_file_path, 'w'))
		print(f'save test set into {test_file_path}')
		
		eval_tabel_file_path = os.path.join(output_dir, str(seed_num), 'eval_tabel', 'eval_tabel.json')
		json.dump(eval_tabel, open(eval_tabel_file_path, 'w'), indent=4)
		print(f'save eval tabel into {eval_tabel_file_path}')

		# further categorize test set into various subsets to test different conditions
		if eval_subsets:
			# unseen verbs v.s. seen verbs
			test_seen_verbs_split, test_unseen_verbs_split = split_seen_unseen_verbs(test_split, seen_verbs)

			test_seen = os.path.join(output_dir, str(seed_num), 'test', 'test_seen_verb.json')
			json.dump(test_seen_verbs_split, open(test_seen, 'w'))

			test_unseen = os.path.join(output_dir, str(seed_num), 'test', 'test_unseen_verb.json')
			json.dump(test_unseen_verbs_split, open(test_unseen, 'w'))

			# top rank verbs v.s. low rank verbs
			test_top_classes, test_low_classes = split_rank(test_split, frame_verb_stats, top_low=[50, -50],
															overall=True)

			test_top = os.path.join(output_dir, str(seed_num), 'test', 'test_top50.json')
			json.dump(test_top_classes, open(test_top, 'w'))

			test_low = os.path.join(output_dir, str(seed_num), 'test', 'test_low50.json')
			json.dump(test_low_classes, open(test_low, 'w'))

			# unseen vids v.s. seen vids
			unseen_vids = test_vids - train_vids
			seen_vids = test_vids - unseen_vids
			print(f'test set seen vids: {len(seen_vids)}')
			print(f'test set unseen vids: {len(unseen_vids)}')

			test_seen_vids_split, test_unseen_vids_split = split_seen_unseen_vids(test_split, seen_vids)

			test_seen_vids = os.path.join(output_dir, str(seed_num), 'test', 'test_seen_vids.json')
			json.dump(test_seen_vids_split, open(test_seen_vids, 'w'))

			test_unseen_vids = os.path.join(output_dir, str(seed_num), 'test', 'test_unseen_vids.json')
			json.dump(test_unseen_vids_split, open(test_unseen_vids, 'w'))
