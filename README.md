# MVDINET
A Novel Multi-level Enzyme Function Predictor with Multi-view Deep Interactive Learning

# Dependency
torch 1.6.0 sklearn python 3.6 

# Content
./MVDINET: main code of MVDINET
./data: the dataset of MVDINET-MCEC, including enzymeAndNoEznyme,enzyme and isoform datasets.

# Usage 
python main.py

# step1:
split train and test data using 5-fold cross-validation

# step2:
Extracting the shallow multi-view data
1）RaptorX Property
2）PSI-BLAST
3）hmmer

# step3:
we take the shallow multi-view data in main code. 


