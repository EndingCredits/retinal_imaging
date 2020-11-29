import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
args = parser.parse_args()

phenotype_map = {
    "cone-rod": [ "ABCA4", "DRAM2", "C21ORF2", "CACNA1F", "GUCA1A" ],
    "rod-cone RP": [ "USH2A", "USH2C", "PRPH2", "ARHGEF18", "BBS1",
                  "CDH23", "CDHR1", "CERKL", "CNGB1", "EYS",
                  "IMPG2", "MERTK", "MYO7A", "NR2E3", "PDE6A",
                  "PDE6B", "PROML1", "PRPF31", "RHO", "RP1",
                  "RP1L1", "RP2", "RPGR", "SNRNP200" ],
    "LCA": [ "RPE65", "RDH12", "CEP290", "GUCY2D", "IQCB1",
             "RPGRIP1", "TULP1", "NMNAT1" ],
    "achromatopsia": [ "CNGB3", "CRB1", "CRX", "CNGA3", "PDE6C" ],
    "choroideremia": [ "CHM" ],
    "vitelliform MD": [ "BEST1" ],
    "retinoschisis": [ "RS1" ],
    "sorsby": [ "TIMP3" ],
    "unknown": [ "AHI1", "MFSD8", "EFEMP1", "PNPLA6" ]
}

import pandas as pd
df = pd.read_csv(args.input_csv)

#print(df.head())

phenotypes = []
not_found = []
for gene in df['gene']:
    for phenotype, genes in phenotype_map.items():
        if gene in genes:
            phenotypes.append(phenotype)
            #print("Found " + phenotype + " for gene " + gene)
            break
    else:
        raise Exception("No phenotype for gene " + gene + " found")
        #phenotypes.append("unknown")
        #not_found.append(gene)
        

#print(not_found)
df['phenotype'] = phenotypes

#print(df.head())

df.to_csv( args.input_csv[:-4] + '_phenotype.csv', index=False)


