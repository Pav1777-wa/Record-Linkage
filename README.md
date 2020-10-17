# Record-Linkage
Record Linkage refers to the process of linking records from different data sources in the absence of unique identifiers. The purpose of 
record linkage is to improve data quality, integrity and to allow the existing data sources to be reused for the new study. It can 
provide new research opportunities as integrated data has more information as compared to a single dataset.
This project implements the record linkage process in python using various data manipulation/processing libraries.
The record linkage process consists of following five steps:

1. Data Preprocessing
2. Blocking
3. Record Pair Comparison
4. Classification
5. Evaluation

I will try to briefly cover the main purpose of each step in the record linkage process.

Data Preprocessing refers to the process of cleaning and standardizing data. This step is considered as the most important as its being said "garbage in garbage out". We need to make sure that our data is free of impurities and properly standardized before moving on to the next step.

Blocking means dividing the datasets into blocks on the basis of certain conditions. There are various blocking techniques such as traditional blocking technique,Nearest Neighbour technique.

To determine the overall similarity between the record pairs, that were generated in the blocking step, a detailed comparison is needed to be done. For each record
pair, we compare all the possible attributes that result in a vector of similarity values for each pair. These vectors are known as Feature/Comparison Vectors. There are three basic areas of comparison, numeric matching, phonetic matching and pattern-based matching which further consists of character-based and tokenbased matching techniques.

The record pairs that were generated in the blocking step and compared in the comparison step are then classified into matches and non-matches. The classification methods can be supervised and unsupervised. The classification approach can be unsupervised and supervised. The unsupervised approach classifies pairs or records based on the similarity between
them in the absence of the training data.
