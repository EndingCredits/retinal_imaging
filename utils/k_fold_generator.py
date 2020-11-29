import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
parser.add_argument('--override', action='store_true', help='Override saved fold (if applicable)')
parser.add_argument('--num-folds', type=int, default=5, help='Number of folds to create')
parser.add_argument('--id-column', type=str, help='Column header for patient ids (leave blank to use random assignment)')
args = parser.parse_args()

import pandas as pd
df = pd.read_csv(args.input_csv)

if not 'fold' in df.columns or args.override:
    import numpy as np
    
    np.random.seed(123)
    np.random.shuffle(indices)
    
    if args.id_column:
        from collections import defaultdict
        
        patient_fold = dict()
        fold_count = defaultdict( lambda: [ 0 ] * args.num_folds )
        folds = np.zeros(len(df)).astype(int)
        
        indices = np.arange(len(df))
        for i in indices:
            patient_id = df[args.id_column][i]
            gene = df['gene'][i]

            if patient_fold.get(patient_id) is None:
                # Assign fold based on fold with lowest num of images
                minvals = np.where(fold_count[gene] == np.min(fold_count[gene]))[0]
                fold = np.random.choice(minvals)
                patient_fold[patient_id] = fold
            else:
                # Use previously stored fold for that patient
                fold = patient_fold.get(patient_id)

            # Update fold counts
            fold_count[gene][fold] += 1
            folds[i] = fold
    else:
        folds = np.repeat(np.arange(args.num_folds), len(df) // args.num_folds + 1)
        np.random.shuffle(folds)
        folds = folds[:len(df)]
    
    df['fold'] = folds
    df.to_csv( args.input_csv[:-4] + '_folds.csv', index=False)
    
for fold in df['fold'].unique():
    #TODO: Make it so we ignore elements with fold > num_folds
    
    df[df['fold'] != fold].to_csv( args.input_csv[:-4] + '_train_{}.csv'.format(fold), index=False)
    df[df['fold'] == fold].to_csv( args.input_csv[:-4] + '_val_{}.csv'.format(fold), index=False)

