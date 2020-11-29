import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset_csv', nargs='+')
parser.add_argument('--data-path', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--mapping-file', type=str)
#parser.add_argument('--num-folds', type=int, default=5, help='Number of folds to create')
args = parser.parse_args()

import glob
import os
from pathlib import Path
import pandas as pd

if args.data_path is None:
	args.data_path = os.getcwd()

files = []
for f in args.dataset_csv:
	files.extend(glob.glob(f))

for f in files:
	df = pd.read_csv(f)

	if args.mapping_file:
		mapping = pd.read_csv(args.mapping_file)

	new_paths = []
	for pth in df["file.path"]:

		pth = pth.replace("/media/pontikos_nas/Data/NikolasPontikos/IRD/MEH/", "")

		if args.mapping_file:
			hosnum = int(pth.split('/')[0])
			new_num = mapping['new_id'][mapping['hosnum'] == hosnum].values[0]

			pth = pth.replace(str(hosnum), str(new_num), 1)

		pth = os.path.join(args.data_path, Path(pth))
		new_paths.append(pth)

	df["file.path"] = new_paths

	if args.output is None:
		df.to_csv(f, index=False)
	else:
		df.to_csv(args.output, index=False)
		break


