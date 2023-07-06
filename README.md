# Quick Start:
- Contact me to get already preprocessed files.
- If you want to customize the creation process, find the below instruction for your information.

# To Reproduce the CAE Dataset Creation Steps:
## Before Start:
Get the relevant resources:
- request **fndata-1.7** from https://framenet.icsi.berkeley.edu/ and save it under the directory **result_verbs/framenet/**.
- download the imSitu annotations from https://github.com/my89/imSitu and save them under  **result_verbs/**.
- clone the repository of **verbnet** from https://github.com/cu-clear/verbnet under **result_verbs/**
- clone the repository of **semlink** from https://github.com/cu-clear/semlink/tree/master under **result_verbs/**.
- get the relevant files of HowTo100M from https://www.di.ens.fr/willow/research/howto100m/: \
  (1) HowTo100M_v1.csv (2) raw_caption_superclean.json and save under **meta_info/HowTo100M**.

## Step 1: Get a list of sure/unsure result verbs
An ideal result verb should possess two properties: (1) visualness (2) effect-causing
```bash
$ cd result_verbs
$ python get_result_verbs.py
```
It should produce two JSON files: (1) sure_result_verbs.json and (2) unsure_result_verbs.json. \
sure_result_verbs.json will be used for the following steps.

## Step 2: Get relevant video clips from HowTo100M to derive the CAE dataset
```bash
python prepare_cae.py --meta_file meta_info/HowTo100M/HowTo100M_v1.csv \
--vids_file meta_info/rank15_result_verbs_downloaded_vids.txt \
--subtitles meta_info/HowTo100M/raw_caption_superclean.json \
--result_verbs meta_info/sure_result_verbs.json \
--concrete_word_file meta_info/Concreteness_ratings_Brysbaert_et_al_BRM.txt \
--categories arts,cars,computers,education,family,food,health,hobbies,holidays,home,personal,pets,sports \
--cache_dir subtitles/domain_cache \
--output_dir subtitles
--process all
```
After running the above code, you should see the following folder structure:
[TODO] Write a script for producing cae.json
```
    ├── subtitles
        ├── domain_cache
            ├── arts
            ├──...
        ├── single_result_verbs_video_clips.json
        ├── single_result_verbs_video_clips_by_vid.json
        ├── single_frames_verbs_stats.json
        ├── single_verbs_nouns_stats.json
        ├── multiple_result_verbs_video_clips.json
        └── multiple_result_verbs_video_clips_by_vid.json
```
- single_result_verbs_video_clips.json looks like this:
```
"Building":{"make":{"arts":{"roigpbZ6Dpc":
[{"vid":"roigpbZ6Dpc","vid seg":38,"time stamp":"1:30:1:36","caption":"you so to start off we're going to make","domain":"arts","frames":["Building"],"verb":"make","nouns":[]},
 {"vid":"roigpbZ6Dpc","vid seg":73,"time stamp":"3:18:3:22","caption":"make this hot cocoa where you want to","domain":"arts","frames": ["Building"],"verb":"make","nouns":["cocoa"]},
...
```
- single_result_verbs_video_clips_by_vid.json is structured along a unique video id:
```
{"roigpbZ6Dpc":
  {"38":{"vid":"roigpbZ6Dpc","vid seg":38,"time stamp":"1:30:1:36","caption":"you so to start off we're going to make","domain":"arts","verbs":["make"],"all_frames":  [["Building"]],"all_nouns":[[]]},
   "73":{"vid":"roigpbZ6Dpc","vid seg":73,"time stamp":"3:18:3:22","caption":"make this hot cocoa where you want to","domain":"arts","verbs":["make"],"all_frames":[["Building"]],"all_nouns":[["cocoa"]]}
  ...
```
- For additional statistic information:
  1. single_frames_verbs_stats.json: video clip counts by unique video clip id across verbs and video domains.
  2. single_verbs_nouns_stats.json: (verb, noun) co-occurrence statistics across verbs and video domains.
     
- For video clips containing **multiple result verbs**, check:
  1. multiple_result_verbs_video_clips.json
  2. multiple_result_verbs_video_clips_by_vid.json

Note: single_result_verbs_video_clips.json and single_frames_verbs_stats.json will be used for the following steps. \

## Step 3: Split the CAE dataset into train/val/test set 
```bash
python split_cae.py --video_clips subtitles/single_result_verbs_video_clips.json \
--frame_verb_stats subtitles/single_frames_verbs_stats.json \
--fixed_seen_verb_list meta_info/kinectics400_joint_verb_labels.txt \
--seeds '42' \
--categories arts,cars,computers,education,family,food,health,hobbies,holidays,home,personal,pets,sports \
--output_dir single_result_verb_exp
```

After running the above code, you should see the following folder structure:
```
    ├── single_result_verb_exp
    │   ├── 42
    │   │    ├── eval_table
    │   │    │    └── eval_table.json
    │   │    ├── train
    │   │    │    └── train.json
    │   │    ├── val
    │   │    │    └── val.json
    │   │    ├── test
                └── test.json
```
