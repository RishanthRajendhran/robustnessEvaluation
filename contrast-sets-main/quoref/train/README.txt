Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning
----------------------------------------------------------------------------------------

This directory contains the Quoref dataset split into train and dev sets. The
test set is not included in this package. The format of each json file is the
same as that of SQuAD, except that the "answers" field can be a list of
multiple spans.

Please read the paper for more details:
https://www.semanticscholar.org/paper/Quoref%3A-A-Reading-Comprehension-Dataset-with-Dasigi-Liu/315642b4698f327ce1f7a76e8973f34f9b54cecd

# Changes in v0.2 #

The version of the dataset in this directory is 0.2, released in June 2021. We
discovered that the indices of answers in version 0.1 of the dataset were off
in a small number of instances (less than 2%) due to unicode processing issues.
Those issues were fixed in this version.
