# CellEmb
Cell line embedding through contrastive learning using drug response correlations.

## Requisite

'''
pytorch = 2.1.0
numpy = 1.26.0
pandas = 2.1.3
tqdm = 4.66.1
Python = 3.10
'''

## Embedding Process

```
python main.py 
```

Starts embeeding process based on the datafiles listed in main.py. Five previously computed embeddings used in the study is located under results directory, and is available for use.

## Creating a Custom Dataset for Embedding

Dataset should be listed as a 4 column tab-delimited text file with first two columns of DepMap cell line IDs, followed by the similarity measure (PCC), and the number of samples used to calculate the similarity. 

The model uses the gene expression provided by the DepMap consortium. Only cell lines with gene expression can be embedded. 

Once prepared, add the filepath to ```pair_pccs``` list within the ```main()``` function of ```main.py```.

## Running the Downstream Task
```
downstream.py [single|combination] [datafile]
```
This process starts the training process for provided dataset.


## Custom Dataset for Downstream Task

Drug embedding for all drugs used are contained in ```data/ChemBERTa_embedded_smiles.pkl```. To create a custom database, python ```dict``` object with pubchem id as key and ChemBERTa embedded tensor as values. 