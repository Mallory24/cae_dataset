import json
import re
import os
import argparse
import sys


import random
import csv
import tqdm
from collections import Counter

import spacy
from spacy.tokens import DocBin
nlp = spacy.load("en_core_web_lg")
import pprint


def nlp_prep(nlp, all_captions, workers=8):
	# aggregate captions across videos for a more efficient spacy nlp preprocessing
	return list(nlp.pipe(all_captions, disable=["ner"], batch_size=64, n_process=workers))


def get_category_vids(meta_file, vids_file):
	# data format: video_id,category_1,category_2,rank,task_id
	with open(vids_file, "r") as f:
		vids = f.readlines()
		selected_vids = [vid.strip("\n") for vid in vids]
		
	category_coarse = dict()
	category_fine = dict()
	all_vids = set()
	with open(meta_file, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count > 0:
				vid = row[0]
				if vid not in selected_vids:
					continue
				if row[1]:
					coarse_cat = row[1].split()[0].lower()  # shorter_name
					fine_cat = row[2].lower()
					fine_cat = coarse_cat + "_" + fine_cat
					if coarse_cat not in category_coarse:
						category_coarse[coarse_cat] = []
					if fine_cat not in category_fine:
						category_fine[fine_cat] = []

					if vid not in all_vids:
						category_coarse[coarse_cat].append(vid)
						category_fine[fine_cat].append(vid)
						all_vids.add(vid)

			line_count += 1
	return category_coarse, category_fine


def spacy_preprocess(category_vids, subtitle_data, cache_dir, partition_size=600000, used_categories=None, useRawCaptions=True):
	for c in category_vids.keys():
		if c in used_categories:
			category_dir = os.path.join(cache_dir, c)
			if not os.path.exists(category_dir):
				os.makedirs(category_dir)

			vids = category_vids[c]
			all_subs = []
			for v in vids:
				sub = [str(t) for t in subtitle_data[v]["text"]]
				all_subs += sub

			print(f"domain {c}: {len(vids)} videos")
			print(f"domain {c}: {len(all_subs)} subtitles")

			for idx, i in enumerate(range(0, len(all_subs), partition_size)):
				doc_bin = DocBin()
				if useRawCaptions:
					saved_path = os.path.join(category_dir, f'{c}_raw_{idx}.spacy')
				else:
					saved_path = os.path.join(category_dir, f'{c}_{idx}.spacy')
				if os.path.exists(saved_path):
					continue
				else:
					seg_subtitles = nlp_prep(nlp, all_subs[i: i + partition_size], workers=4)
					for seg_sub in seg_subtitles:
						doc_bin.add(seg_sub)
					doc_bin.to_disk(saved_path)
			print(f'done spacy preprocessing for domain {c} and save into {category_dir}.')


def convert_sec2min(seconds):
	min, sec = divmod(seconds, 60)
	return f"{int(min)}:{int(sec):02d}"


def convert_min2sec(min, sec):
	return int(min) * 60 + int(sec)


def check_nouns_via_dep(subtitle):
	nouns = []
	for t in subtitle:
		if t.dep_ in ["dobj", "pobj"]:
			nouns.append(t.lemma_)
	return nouns


def get_concrete_words(concrete_nouns_file):
	with open(concrete_nouns_file,'r') as f:
		columns=[]
		for l in f.readlines():
			l = l.split()
			if columns:
				if len(l) > 9:
					columns[0].append(l[0])
					columns[0].append(l[1])
					for i, value in enumerate(l[2:]):
						columns[i+1].append(value)
				else:
					for i, value in enumerate(l):
						columns[i].append(value)
			else:
				columns = [[value] for value in l]

	concreteness = {c[0] : c[1:] for c in columns}
	words = [w.lower() for w in concreteness["Word"]]
	con_score = [s for s in concreteness["Conc.M"]]

	concrete_words = []
	for w, c in zip(words, con_score):
		if 4 < float(c) <= 5:
			concrete_words.append(w)
	return concrete_words


def count_frame_verb(frames_video_clips):
	stats = {}
	for F, V in frames_video_clips.items():
		if F not in stats:
			stats[F] = {"verbs": {}}

		for v, D in V.items():
			stats[F]["verbs"][v] = {}
			for d, vid_clips_by_vid in D.items():
				stats[F]["verbs"][v][d] = {}
				for vid in vid_clips_by_vid:
					stats[F]["verbs"][v][d][vid] = len(vid_clips_by_vid[vid])
	return stats


def count_frame_verb_noun(frames_video_clips, concrete_words):
	stats = {}
	for F, V in frames_video_clips.items():
		if F not in stats:
			stats[F] = {"verbs": {}}

		for v, D in V.items():
			stats[F]["verbs"][v] = {}
			for d, vid_clips_by_vid in D.items():
				stats[F]["verbs"][v][d] = {}
				for vid, vid_clips in vid_clips_by_vid.items():
					for vid_clip in vid_clips:
						for n in vid_clip["nouns"]:
							if n in concrete_words:
								if n not in stats[F]["verbs"][v][d]:
									stats[F]["verbs"][v][d][n] = 0
								stats[F]["verbs"][v][d][n] += 1
	return stats


def extract_result_verbs(category, verbs_annotation, category_vids, preprocessed_captions, data):
	frames_video_clips = {}
	c = 0   # boundary of video
	for v in category_vids:
		for i in range(len(data[v]["text"])):
			p = preprocessed_captions[i+c]
			s = convert_sec2min(data[v]["start"][i])
			e = convert_sec2min(data[v]["end"][i])
			time = s + ":" + e
			for idx, t in enumerate(p):
				if t.lemma_ in verbs_annotation.keys() and t.pos_ == "VERB":
					nouns = check_nouns_via_dep(p)
					mapped_frames = verbs_annotation.get(t.lemma_)["frames"]

					for f in mapped_frames:
						if f not in frames_video_clips:
							frames_video_clips[f] = {}
						if t.lemma_ not in frames_video_clips[f]:
							frames_video_clips[f][t.lemma_] = []
						frames_video_clips[f][t.lemma_].append({"vid": v, "time stamp": time,
						                                        "verb": t.lemma_,
																"frames": mapped_frames, "nouns": nouns,
																"caption": data[v]["text"][i],
																"vid seg": i, "domain": category})
		c += len(data[v]["text"])
	return frames_video_clips


# Re-structure into vid based data format
def re_format(data):
	"""

	:param data: frame based format: {F: {verb: {domain: {vid: [{vid_seg_info1}, {vid_seg_info2}]}}}
	:return: video id based format: {vid: seg1: {seg1_info}, seg2: {seg1_info}}
	"""
	result = {}
	all_verbs = set()
	for F, verb_dicts in data.items():    # {'make': {'arts': {vid: {info}}, 'create': ... }
		for verb, d_dict in verb_dicts.items():    # d_dict => {'arts': {vid: {info}}, 'cars':  {vid: {info}} ...}
			if verb in all_verbs:   # since multiple frames can be mapped to the same verb
				continue
			else:
				all_verbs.add(verb)

			for d, vid_dict in d_dict.items():
				for vid, vid_clips in vid_dict.items():
					if vid not in result:
						result[vid] = {}

					for vid_clip in vid_clips:
						vid_seg = vid_clip["vid seg"]
						if vid_seg in result[vid]:
							# result verbs are located in the same video segment
							result[vid][vid_seg]["verbs"].append(vid_clip["verb"])
							result[vid][vid_seg]["all_frames"].append(vid_clip["frames"])
							result[vid][vid_seg]["all_nouns"].append(vid_clip["nouns"])
						else:
							new_vid_clip = {}
							new_vid_clip["vid"] = vid_clip["vid"]
							new_vid_clip["vid seg"] = vid_clip["vid seg"]
							new_vid_clip["time stamp"] = vid_clip["time stamp"]
							new_vid_clip["caption"] = vid_clip["caption"]
							new_vid_clip["domain"] = vid_clip["domain"]
							
							new_vid_clip["verbs"] = []
							new_vid_clip["verbs"].append(vid_clip["verb"])
							new_vid_clip["all_frames"] = []
							new_vid_clip["all_frames"].append(vid_clip["frames"])
							new_vid_clip["all_nouns"] = []
							new_vid_clip["all_nouns"].append(vid_clip["nouns"])
							result[vid][vid_seg] = new_vid_clip

	return result


def get_vid_seg_ids_by_type(data):
	"""

	:param data: video id based format: {vid: seg1: {seg1_info}, seg2: {seg1_info}}
	:return: two sets of video segment ids of different types
			(single-action v.s. multiple-actions)
	"""
	single_action_vid_seg_ids = set()
	multiple_actions_vid_seg_ids = set()
	# TODO: address the case where multiple verbs are of single class
	
	for vid in data.keys():
		for seg_id in data[vid].keys():
			verbs = data[vid][seg_id]["verbs"]
			if len(verbs) > 1:
				vid_seg_id = vid + "_" + str(seg_id)
				multiple_actions_vid_seg_ids.add(vid_seg_id)
			else:
				vid_seg_id = vid + "_" + str(seg_id)
				single_action_vid_seg_ids.add(vid_seg_id)
	
	return single_action_vid_seg_ids, multiple_actions_vid_seg_ids


def group_by_action_types(data, single_vid_segs_set, multiple_vid_segs_set):
	"""

	:param data: frame based format
			single_vid_segs_set: video segment ids of single actions
			multiple_vid_segs_set: video segment ids of multiple actions
	:return: two frame based formats
			(single-action v.s. multiple-actions)
	"""
	
	single_data = {}
	multiple_data = {}
	
	for F, verb_dicts in data.items():  # verb_dicts => {'make': {'arts': {vid: {info}}, 'create': ... }
		if F not in single_data:
			single_data[F] = {}
		if F not in multiple_data:
			multiple_data[F] = {}
		
		for verb, d_dict in verb_dicts.items():  # d_dict => {'arts': {vid: {info}}, 'cars':  {vid: {info}} ...}
			if verb not in single_data[F]:
				single_data[F][verb] = {}
			if verb not in multiple_data[F]:
				multiple_data[F][verb] = {}
			
			for d, vid_dict in d_dict.items():
				if d not in single_data[F][verb]:
					single_data[F][verb][d] = {}
				if d not in multiple_data[F][verb]:
					multiple_data[F][verb][d] = {}
				
				for vid, vid_clips in vid_dict.items():
					for vid_clip in vid_clips:
						vid_seg_id = vid + "_" + str(vid_clip["vid seg"])
						if vid_seg_id in single_vid_segs_set:
							if vid not in single_data[F][verb][d]:
								single_data[F][verb][d][vid] = []
							single_data[F][verb][d][vid].append(vid_clip)
						elif vid_seg_id in multiple_vid_segs_set:
							if vid not in multiple_data[F][verb][d]:
								multiple_data[F][verb][d][vid] = []
							multiple_data[F][verb][d][vid].append(vid_clip)
	
	return single_data, multiple_data


def remove_consecutive_video_segment(data):
	"""

	:param data: frame based format
	:return: frame based formats without consecutive video segments
	"""
	result = {}
	for vid in data.keys():
		if vid not in result:
			result[vid] = {}
			
		vid_seg_list = list(data[vid].keys())
		vid_seg_list = [int(vid_seg) for vid_seg in vid_seg_list]
		
		time_stamp_list = [vid_seg_info["time stamp"] for vid_seg_info in data[vid].values()]
		
		vid_seg_list, time_stamp_list = zip(*sorted(zip(vid_seg_list, time_stamp_list)))
		start_time_stamp_list = [convert_min2sec(time_stamp.split(":")[0], time_stamp.split(":")[1])
		                         for time_stamp in time_stamp_list]
		end_time_stamp_list = [convert_min2sec(time_stamp.split(":")[-2], time_stamp.split(":")[-1])
		                       for time_stamp in time_stamp_list]
		
		prev_end_time_stamp = 0
		for idx, start_time_stamp in enumerate(start_time_stamp_list):
			if idx > 0:
				if start_time_stamp - prev_end_time_stamp >= 5:
					# only keep non-consecutive vid segs that have temporal difference of at least 5 secs
					result[vid][str(vid_seg_list[idx])] = data[vid][str(vid_seg_list[idx])]
					prev_end_time_stamp = end_time_stamp_list[idx]
			else:
				result[vid][str(vid_seg_list[idx])] = data[vid][str(vid_seg_list[idx])]
				prev_end_time_stamp = end_time_stamp_list[idx]
	return result
	
	
def extract_result_verbs_vid_clips(result_verbs, category_vids, subtitle_data, cache_dir, output_dir, used_categories=None, concrete_word_file=None):
	all_result_verbs_vid_clips = {}
	concrete_words = get_concrete_words(concrete_word_file)

	# extract video clips that contain result verbs
	for c in used_categories:
		vids = category_vids[c]
		category_dir = os.path.join(cache_dir, c)
		preprocessed_paths = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if
							  os.path.isfile(os.path.join(category_dir, f)) and f.endswith(".spacy")]
		preprocessed_order = [int(p.split("_")[-1].strip('.spacy')) for p in preprocessed_paths]
		preprocessed_paths = [x for _, x in sorted(zip(preprocessed_order, preprocessed_paths))]

		all_subs = []
		for v in vids:
			sub = [str(t) for t in subtitle_data[v]["text"]]
			all_subs += sub

		all_annot_subs = []
		for prep_path in preprocessed_paths:
			doc_bin = DocBin().from_disk(prep_path)
			subs = list(doc_bin.get_docs(nlp.vocab))
			all_annot_subs += subs

		assert len(all_subs) == len(all_annot_subs)

		domain_frames_video_clips = extract_result_verbs(c, result_verbs, vids, all_annot_subs, subtitle_data)
		for F in domain_frames_video_clips.keys():
			if F not in all_result_verbs_vid_clips:
				all_result_verbs_vid_clips[F] = {}

			for v, vid_clips in domain_frames_video_clips[F].items():
				vid_clips_by_vid = {}
				for vid_clip in vid_clips:
					if vid_clip["vid"] not in vid_clips_by_vid:
						vid_clips_by_vid[vid_clip["vid"]] = []
					vid_clips_by_vid[vid_clip["vid"]].append(vid_clip)
				if v not in all_result_verbs_vid_clips[F]:
					all_result_verbs_vid_clips[F][v] = {}
				all_result_verbs_vid_clips[F][v][c] = vid_clips_by_vid

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		
	
	# re-format by vid
	all_result_verbs_vid_clips_by_vid = re_format(all_result_verbs_vid_clips)

	# remove consecutive video segments
	non_con_vid_clips_by_vid = remove_consecutive_video_segment(all_result_verbs_vid_clips_by_vid)

	# split into single actions v.s. multiple actions
	print(f'split into two groups: single action v.s. multiple actions')

	single_vid_seg_ids, multiple_vid_seg_ids = \
		get_vid_seg_ids_by_type(non_con_vid_clips_by_vid)
	
	single_result_verbs_vid_clips, multiple_result_verbs_vid_clips = \
		group_by_action_types(all_result_verbs_vid_clips, single_vid_seg_ids, multiple_vid_seg_ids)
	
	single_result_verbs_vid_clips_path = os.path.join(output_dir, "single_result_verbs_video_clips.json")
	json.dump(single_result_verbs_vid_clips, open(single_result_verbs_vid_clips_path, 'w'))
	
	single_result_verbs_vid_clips_by_vid = re_format(single_result_verbs_vid_clips)
	single_result_verbs_vid_clips_by_vid_path = os.path.join(output_dir, "single_result_verbs_video_clips_by_vid.json")
	json.dump(single_result_verbs_vid_clips_by_vid, open(single_result_verbs_vid_clips_by_vid_path, 'w'))
	
	multiple_result_verbs_vid_clips_path = os.path.join(output_dir, "multiple_result_verbs_video_clips.json")
	json.dump(multiple_result_verbs_vid_clips, open(multiple_result_verbs_vid_clips_path, 'w'))

	multiple_result_verbs_vid_clips_by_vid = re_format(multiple_result_verbs_vid_clips)
	multiple_result_verbs_vid_clips_by_vid_path = os.path.join(output_dir, "multiple_result_verbs_video_clips_by_vid.json")
	json.dump(multiple_result_verbs_vid_clips_by_vid, open(multiple_result_verbs_vid_clips_by_vid_path, 'w'))
	
	# counting statistics for single verb action only
	frame_verb_stats = count_frame_verb(single_result_verbs_vid_clips)
	verb_noun_stats = count_frame_verb_noun(single_result_verbs_vid_clips, concrete_words)

	frame_verb_stats_path = os.path.join(output_dir, "single_frames_verbs_stats.json")
	json.dump(frame_verb_stats, open(frame_verb_stats_path, 'w'), indent=4)
	print(f'done calculating frame verb statistics for single-action cases and save into {frame_verb_stats_path}')
	
	verbs_nouns_stats_path = os.path.join(output_dir, "single_verbs_nouns_stats.json")
	json.dump(verb_noun_stats, open(verbs_nouns_stats_path, 'w'), indent=4)
	print(f'done calculating verb noun statistics for single-action cases and save into {verbs_nouns_stats_path}')


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Extracting video clips that contain result verb')
	parser.add_argument('--device', default="0", type=str, help="select gpu device or cpu")

	parser.add_argument("--meta_file", type=str, default="HowTo100M_v1.csv", required=True,
						help="HowTo100M meta file")
	parser.add_argument("--vids_file", type=str, default="rank15_result_verbs_downloaded_vids.txt", required=True,
						help="Top 15 viewed videos per task id")
	parser.add_argument("--subtitles", type=str, default="raw_caption_superclean.json", required=True,
						help="subtitles dataset")
	parser.add_argument("--concrete_word_file", type=str, default="Concreteness_ratings_Brysbaert_et_al_BRM.txt",
						required=True, help="concreteness ratings file")
	parser.add_argument("--result_verbs", type=str, default="result_verbs/sure_result_verbs.json", required=True,
						help="result verbs list")
	parser.add_argument('--categories', type=str, default="family,food,hobbies", required=True,
						help="select video categories for preprocessing")
	parser.add_argument('--cache_dir', type=str, default="domain_cache", required=True,
						help="directory of spacy preprocessed cache")
	parser.add_argument('--output_dir', type=str, default="final_dataset", required=True,
						help="output directory of preprocessed dataset")
	parser.add_argument("--process", default='post', type=str, required=False, choices=['pre', 'post', 'all'],
						help="define which process to execute")

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	meta_file = args.meta_file
	vids_file = args.vids_file
	subtitles = args.subtitles
	cache_dir = args.cache_dir
	output_dir = args.output_dir
	used_categories = args.categories.split(",")
	result_verbs = json.load(open(args.result_verbs))

	category_coarse, category_fine = get_category_vids(meta_file, vids_file)
	subtitle_data = json.load(open(subtitles, 'r'))

	if args.process == "pre":
		spacy_preprocess(category_coarse, subtitle_data, cache_dir, used_categories=used_categories)
	elif args.process == "post":
		extract_result_verbs_vid_clips(result_verbs, category_coarse, subtitle_data, cache_dir, output_dir,
		                               used_categories=used_categories,
		                               concrete_word_file=args.concrete_word_file)

	elif args.process == "all":
		spacy_preprocess(category_coarse, subtitle_data, cache_dir, used_categories=used_categories)
		extract_result_verbs_vid_clips(result_verbs, category_coarse, subtitle_data, cache_dir, output_dir,
		                               used_categories=used_categories,
		                               concrete_word_file=args.concrete_word_file)
