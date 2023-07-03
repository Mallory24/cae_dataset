from verbnet.api.verbnet import *
import json
import spacy
import xml.etree.ElementTree as ET
nlp = spacy.load("en_core_web_lg")

# general helper functions
def json_load(path):
	with open(path, 'r') as json_file:
		return json.load(json_file)

def json_dump(file, path):
	with open(path, 'w') as json_file:
		json.dump(file, json_file, indent=4, sort_keys=True)


# VerbNet helper functions
def has_role(verb_value, role_type):
	if len([r for r in verb_value.themroles if r.role_type == role_type]) != 0:
		return True
	return False

def roles_inherited(verb_value):
	if len(verb_value.themroles) == 0:
		return True
	return False

def is_CoS_verbs(verb_key):
	if verb_key.split("-")[1].startswith("45"):
		return True
	return False

def is_solid(verb_value):
	for r in verb_value.themroles:
		if r.role_type == "Patient":
			for sel in r.sel_restrictions:
				if sel in [['+', 'concrete'], ['+', 'solid']]:
					return True
				elif sel in ['concrete', 'solid']:
					if "-" not in r.sel_restrictions:
						return True
			return False

def add_members(result_verbs_annotation, values, fn_mappings, visualness, cause_result):
	for n in values.names:
		vn = values.numerical_ID + "-" + n
		if fn_mappings.get(vn) is not None:
			result_verbs_annotation.update({vn: {"visualness": visualness,
												 "frames": fn_mappings.get(vn),
												 "cause result": cause_result}})
	for sub in values.subclasses:
		if roles_inherited(sub):
			result_verbs_annotation = add_members(result_verbs_annotation, sub, fn_mappings, visualness=visualness, cause_result=cause_result)
	return result_verbs_annotation


# imSitu helper function
def extract_frames_via_valid_role_order(verb_frames, position, bad_roles, LU_Fs):
	verb_frames_output = {}
	imSitu_defined_roles = set()
	for k, v in verb_frames.items():
		lemma = nlp(k)[0].lemma_
		roles = verb_frames[k]["order"]
		if len(roles) > 1:
			if roles[position] not in bad_roles:
				# verb_frames_output.update({lemma: {"visualness": "yes", "frames": list(v["framenet"]), "cause result": "unsure"}})
				lu = lemma + ".v"
				if lu in LU_Fs: #if the lexical unit is in the current FN1.7 lexicon
					FN_frames = LU_Fs[lu]
					if v["framenet"] in FN_frames: #if the lexical unit - frame mapping is also correct
						verb_frames_output.update({lemma: {"visualness": "yes", "frames": [v["framenet"]], "cause result": "unsure"}})
	return verb_frames_output, imSitu_defined_roles


# FrameNet helper function
def get_LU_frames_mappings(FN_data):
	LU_Fs = {}
	for F in FN_data:
		mytree = ET.parse(F)
		myroot = mytree.getroot()
		frame = myroot.attrib["name"]
		for x in myroot.findall("{http://framenet.icsi.berkeley.edu}lexUnit"):
			lu = x.attrib['name']
			if lu not in LU_Fs:
				LU_Fs[lu] = []
			LU_Fs[lu].append(frame)
	return LU_Fs

def get_frame_FEs_mappings(FN_data):
	F_elements = {}
	for F in FN_data:
		mytree = ET.parse(F)
		myroot = mytree.getroot()
		frame = myroot.attrib["name"]
		F_elements[frame] = []
		for x in myroot.findall("{http://framenet.icsi.berkeley.edu}FE"):
			role = x.attrib['name']
			F_elements[frame].append(role)
	return F_elements

# get FrameNet annotation
frame_dir = "framenet/fndata-1.7/frame"
frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if os.path.isfile(os.path.join(frame_dir, f)) and f.endswith("xml")]
LU_Fs = get_LU_frames_mappings(frame_files)
F_elements = get_frame_FEs_mappings(frame_files)

# simple heuristic to identify result frames
FN_frames_with_result = [k for k, v in F_elements.items() for ele in v if ele in ["Result", "Effect"]]

# leverage VerbNet's annotation to derive potential set of result verbs
vnp = VerbNetParser(directory="verbnet/verbnet3.3")
semlink_path = "semlink/instances/vn-fn2.json"
vn_fn_mappings = json_load(semlink_path)

VN_potential_result_verbs = {}
for k, v in vnp.verb_classes_dict.items():
	if has_role(v, "Patient"):
		if has_role(v, "Agent") and has_role(v, "Result"):
			if is_solid(v):
				VN_potential_result_verbs = add_members(VN_potential_result_verbs, v, vn_fn_mappings, visualness="yes", cause_result="yes")
			else:
				VN_potential_result_verbs = add_members(VN_potential_result_verbs, v, vn_fn_mappings, visualness="unsure", cause_result="yes")
		else:
			if is_solid(v):
				VN_potential_result_verbs = add_members(VN_potential_result_verbs, v, vn_fn_mappings, visualness="yes", cause_result="unsure")

# leverage imSitu's annotation to derive potential set of result verbs
imsitu = json_load("imsitu_annotations/imsitu_space.json")
imsitu_frames = imsitu["verbs"]

bad_roles = ["place", "tool", "location", "manner", "instrument", "listener",
			 "container", "model", "suspect", "victimpart", "addressee",
			 "confronted", "start", 'message', 'skill', 'ailment', 'focus',
			 'resource', 'experiencer', 'phenomenon', 'agentpart', 'coagent',
			 'end', 'recipient', 'audience', 'blow', 'supported', 'interviewee',
			 'destination', 'source', 'carrier', 'entityhelped', 'center', 'reciever', 'event',
			 'naggedperson', 'obstacle', 'stake', 'coparticipant', 'seller', 'performer', 'student',
			 'giver', 'reference', 'adressee', 'competition', 'occasion', 'image', 'coagentpart',
			 'bodypart', 'boringthing', 'victim', 'follower', 'perceiver', 'imitation', 'admired',
			 'chasee', 'undergoer', 'path', 'shelter', 'restrained']

imSitu_potential_result_verbs, imSitu_roles = extract_frames_via_valid_role_order(imsitu_frames, 1, bad_roles, LU_Fs)

# leverage FrameNet's annotation to determine result verbs
for annot in [VN_potential_result_verbs, imSitu_potential_result_verbs]:
	for k, v in annot.items():
		if any([True for f in v["frames"] if f in FN_frames_with_result]):
			v["cause result"]="yes"


# merge VerbNet and imSitu
def add_frame(annotation, frames):
	for f in frames:
		if f not in annotation["frames"]:
			annotation["frames"].append(f)
	return annotation

full_result_verbs_annotation = {}
for annot in [VN_potential_result_verbs, imSitu_potential_result_verbs]:
	for k, v in annot.items():
		if type(k) is str:
			if "-" not in k:
				lemma = k
			else:
				lemma = k.split("-")[-1]

		if lemma not in full_result_verbs_annotation:
			full_result_verbs_annotation[lemma] = v
			full_result_verbs_annotation[lemma]["verb senses"] = []

		# sanity check: is visualness and cause_result in agreement no matter which verb sense it is (--> no)
		if full_result_verbs_annotation[lemma]["visualness"] != v["visualness"]:
			if v["visualness"] == "yes":
				full_result_verbs_annotation[lemma]["visualness"] = "yes"
				full_result_verbs_annotation[lemma] = add_frame(full_result_verbs_annotation[lemma], v["frames"])

		if full_result_verbs_annotation[lemma]["cause result"] != v["cause result"]:
			if v["cause result"] == "yes":
				full_result_verbs_annotation[lemma]["cause result"] = "yes"
				full_result_verbs_annotation[lemma] = add_frame(full_result_verbs_annotation[lemma], v["frames"])

		full_result_verbs_annotation[lemma] = add_frame(full_result_verbs_annotation[lemma], v["frames"])
		full_result_verbs_annotation[lemma]["verb senses"].append(k)


print("VN potential result verbs counts:", len(VN_potential_result_verbs))
print("imSitu potential result verbs counts:", len(imSitu_potential_result_verbs))

# get unsure cases only
unsure_result_verbs_annotation = {}
for k, v in full_result_verbs_annotation.items():
	if any([v["visualness"] != "yes", v["cause result"] != "yes"]):
		unsure_result_verbs_annotation[k] = v

# get sure cases only
sure_result_verbs_annotation = {}
for k, v in full_result_verbs_annotation.items():
	if v["visualness"] == "yes" and v["cause result"] == "yes":
		sure_result_verbs_annotation[k] = v

# how many unique FN frames
FN_Frames = set()
for k, v in full_result_verbs_annotation.items():
	for f in v["frames"]:
		FN_Frames.add(f)

print("Result verbs counts after merging:", len(full_result_verbs_annotation))
print("unsure cases counts:", len(unsure_result_verbs_annotation))
print("sure cases counts:", len(sure_result_verbs_annotation))
print("unique FN frames:", len(FN_Frames))

json_dump(unsure_result_verbs_annotation, "unsure_result_verbs.json")
json_dump(sure_result_verbs_annotation, "sure_result_verbs.json")
