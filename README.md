# CellEmb
Cell line embedding through drug response correlations 

## Requisite

pytorch = 2.1.0
numpy = 1.26.0
pandas = 2.1.3
tqdm = 4.66.1

## Preparing Missing Datafiles
You need to create a python dict that uses PubChem compound ids as key, and the embedded results of ChemBERTa-77M-MTR with canonical SMILES string of the input. 
