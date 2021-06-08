import pandas as pd
import os
import argparse

def merge_query(args):
	data = []
	for line in open(args.query).readlines():
		items = line.split('||')
		key = items[0]
		content = '||'.join(items[1:])
		data.append((key, content))
	
	data = pd.DataFrame(data=data, columns=['key', 'content'])
	data.to_csv('./query.csv', index=False)

def merge_folder(folder, outfile):
	data = []
	for filename in os.listdir(folder):
		key = filename.split('.')[0]
		content = open(folder + filename).read()
		data.append((key, content))
	
	data = pd.DataFrame(data=data, columns=['key', 'content'])
	data.to_csv(outfile, index=False)
	
def merge_pair(infile, outfile, name='case'):
	data = []
	for line in open(infile).readlines():
		line = line.split()
		data.append((line[0], line[2], line[-1]))
		
	data = pd.DataFrame(data=data, columns=['query', name, 'label'])
	data.to_csv(outfile, index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--query', dest='query', type=str, required=True)
	parser.add_argument('--statute', dest='statute', type=str, required=True)
	parser.add_argument('--case', dest='case', type=str, required=True)
	parser.add_argument('--pair_case', dest='pair_case', type=str, required=True)
	parser.add_argument('--pair_statute', dest='pair_statute', type=str, required=True)
	args = parser.parse_args()
	
	# merge query
	merge_query(args)
	# merge case
	merge_folder(args.case, './case.csv')
	# merge statute
	merge_folder(args.statute, './statute.csv')
	# merge pair case
	merge_pair(args.pair_case, './pair_case.csv')
	# merge pair statute
	merge_pair(args.pair_statute, './pair_statute.csv', name='statute')
	
	# merge the cases and statutes
	case = pd.read_csv('./case.csv')
	query = pd.read_csv('./query.csv')
	statute = pd.read_csv('./statute.csv')
	data = case.append(statute)
	data = data.append(query)
	data.to_csv('./doc.csv', index=False)
