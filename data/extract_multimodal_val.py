import os
import json
import pdb
import random
from argparse import ArgumentParser
random.seed(42)


def main(args):
	document = json.load(open(args.mul_doc,'r'))
	val_query = json.load(open(args.mul_val,'r'))


	document_dict = dict()
	for doc in document:
		document_dict[doc['id']] = doc

	val_document = []
	for idx, query in enumerate(val_query):
		new_doc = document_dict[query['id']].copy()
		del document_dict[query['id']]
		query["id"] = idx
		new_doc["id"] = idx
		val_document.append(new_doc)
	
	remaining_doc = list(document_dict.values())
	add_doc = random.sample(remaining_doc,args.val_amount - len(val_document))
	idx = len(val_document)
	for doc in add_doc:
		doc['id'] = idx
		idx += 1
	val_document += add_doc
	

	
	with open(args.mul_val, "w") as outfile:
		json.dump(val_query, outfile, indent = 4)   
	with open(args.output, "w") as outfile:
		json.dump(val_document, outfile, indent = 4)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--mul_doc', default='multimodal_documents.json')
	parser.add_argument('--mul_val', default='multimodal_val_queries.json')
	parser.add_argument('--val_amount', default=10000,type=int)
	parser.add_argument('--output', default="multimodal_val_documents.json")
	args = parser.parse_args()

	print(args)
	main(args)