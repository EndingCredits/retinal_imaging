import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_paths', nargs='+')
parser.add_argument('--output', type=str, default="data.csv")
#parser.add_argument('--override', action='store_true', help='Override saved fold (if applicable)')
#parser.add_argument('--num-folds', type=int, default=5, help='Number of folds to create')
args = parser.parse_args()

import pandas as pd
df = pd.DataFrame(columns=['file.path', 'gene'])

import os
for data_path in args.data_paths:
    #TODO: Add glob-ing
    for d in os.listdir(data_path):
        for f in os.listdir(os.path.join(data_path, d)):
            img_path = os.path.join(os.path.join(data_path, d), f)
            df = df.append({'file.path': img_path, 'gene': d}, ignore_index=True)
   
df.to_csv(args.output, index=False)