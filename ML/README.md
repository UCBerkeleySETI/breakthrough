# Machine Learning with Breakthrough Listen Data

This section holds a repository of machine learning techniques developed for Breakthrough Listen with various implementations and for various use cases. 

## CNNFRB's 
#### Contributed by Yunfan Gerry Zhang| [Contact] (yf.g.zhang@gmail.com)
This is a neural network for fast radio burst detection. This is an inference script for detecting FRB with a neural network. It is designed for use with the Breakthrough Listen data product at the Green Bank Telescope. [Checkout](CNNFRB/README.md)

## DeepSeti_Semi_Supervised
#### Contributed by Peter Ma| [Contact] (https://peterma.ca/)
A novel algorithm designed to detect anomalies for radio telescope data open-sourced by Breakthrough Listen. The goal is to develop a model to search for anomalies that we don't even know of. To achieve this the model implements novel training techniques. [Checkout](DeepSeti_Semi_Supervised/README.md)

## Non Negative Tensor/Matrix Factorization
#### Contributed by Peter Ma| [Contact] (https://peterma.ca/)
A novel algorithm designed to delete RFI that repeats in multiple scans within a cadence. This only works with signals that do not change and evolve across observations. It is good at removing certain artifacts within the scans. [Checkout](NTF_RFI_Isolation/README.md)

## Kepler-analysis
#### Contributed by Pragaash Ponnusamy| [Contact] (https://github.com/pragaashp)
Kepler Analysis of the data using methods of clustering signals in frequency[Checkout](Kepler-analysis/analysis.ipynb)
