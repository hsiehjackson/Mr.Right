import json

def prepare_pretrain_data(files):
	print("\nReading json files")
	image_text_pairs = []
	for f in files:
		print(f"File: {f}",end="\r")
		image_text_pairs += json.load(open(f,'r'))
	return image_text_pairs

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self