# Quick Start:
- [TODO] create conda env
- [TODO] Write the path to the file

# To Reproduce the CAE Dataset Preprocessing Steps:

## Before Start:
Create the relevant resource:
- request **fndata-1.7** from https://framenet.icsi.berkeley.edu/ and save it under the directory **result_verbs/framenet/**.
- download the imSitu annotations from https://github.com/my89/imSitu and save them under  **result_verbs/**.
- clone the repository of **verbnet** from https://github.com/cu-clear/verbnet under **result_verbs/**
- clone the repository of **semlink** from https://github.com/cu-clear/semlink/tree/master under **result_verbs/**.

## Step 1: Get a list of sure/unsure result verbs
An ideal result verb should possess two properties: (1) visualness (2) effect-causing
```bash
$ cd result_verbs
$ python get_result_verbs.py
```
It should produce two JSON files: (1) sure_result_verbs.json (2) unsure_result_verbs.json. \
Save sure_result_verbs.json under **meta_info**
- [TODO] Provide examples of sure/unsure result verbs

## Step 2: Preprocess the HowTo100M subtitles
