import json
import pandas as pd



def json_parser(json_name):
	'''
	param json_name str of filename to open and read in a json file with rink locations 

	return image_list[str] a list of all labeled image file names
	'''
	with open(json_name) as f:
		data = json.load(f)

    # convert json to pandas dataframe
	data = pd.DataFrame(data)

	# remove skipped images
	data = data[data.Label != 'Skip']

	# remove images that don't have updated naming convention
	elim_lst = ["RushPlay_" in frame_name for frame_name in data["External ID"]]
	data = data[elim_lst]

	# N data points we can work with
	N = len(data['External ID'])

	# convert to list of image filenames
	image_list = data['External ID'].tolist()
	print("blahhhh")

	print(data['Label'])
	





	return image_list


