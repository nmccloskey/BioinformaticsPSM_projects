# BioinformaticsPSM_projects
course projects from professional science master's in bioinformatics at Temple University

This repo contains final projects for six courses completed in Temple University's Bioinformatics PSM program.

## Fall 2021

### Biological Models in Python

This project investigated whether machine learning techniques could predict opioid motivation phenotype (stress resistance or sensitivity) from ultrasonic vocalizations of lab rats. Results indicated that call features produced by the MATLAB package DeepSqueak allowed models to accurately classify the behavioral category of the rodent emitting the call, and an update to this project along with a manuscript are underway, to incorporate more data and additional classifiers.

### Genomics

Telomeres are repetitve sequences at the ends of DNA molecules whose functions relate to the aging process and somatic robustness. Motivated by a publication (Weinstein B, Ciszek D. The reserve-capacity hypothesis: evolutionary origins and modern implications of the trade-off between tumor-suppression and tissue-repair. Experimental Gerontology. 2002;37:615-27) questioning the validity of common animal models for certain biomedical investigations based on telomeric disparities between mice and humans, this project aimed to evaluate if laboratory rats possess the same potentially confounding extra-long telomeres, by accessing DNA sequences with Entrez. The results indicate that rats do possess longer telomeres than humans, which suggests that conclusions from studies using rat models to investigate carcinogenicity and tissue damage may be misleading.

## Spring 2022

### Computational Genomics

This project comes from Dr. David Liberles's research group, and the objective is to simulate the evolution of metabolic pathways with kinetic modeling. Uploaded here is the preliminary version, to which there have been extensive updates. The final version of this project and a manuscript are underway.

### Scripting for Sciences and Business

This upload contains example files from computer labs the bulk of whose content comes from the instructor of the course (Dr. Justin Shi), included here to demonstrate the effort and learning involved in their completion. An interactive web page was constructed to interface with various programs and a user login database, involving HTML, CSS, PHP, MySQL, Python, C, Bootstrap, JQuery, and Javascript. Link: https://cis-linux2.temple.edu/~tuf61393/5015/lab11.

## Fall 2022

### Biostatistics

This project contains two parts, both written in R. The first conducted a statistical analysis of behavioral data from a previosly published study I condcuted in the laboratory of Dr. Lynn Kirby (Li C, McCloskey N, Phillips J, Simmons SJ, Kirby LG. CRF-5-HT interactions in the dorsal raphe nucleus and motivation for stress-induced opioid reinstatement. Psychopharmacology (Berl). 2021 Jan;238(1):29-40. doi: 10.1007/s00213-020-05652-3. Epub 2020 Nov 24. PMID: 33231727; PMCID: PMC7796902). The second constructed two Naïve Bayes classifiers, one Gaussian binary and one Bernoulli multi-class, using Kegg databases.

### Machine Learning

Following an exploratory data analysis, this project used 3 Kegg datasets (of different sizes) to evaluate the performance of various binary classifiers trained and tested on each dataset (LSTM neural nets, Random Forests, Gaussian Naïve Bayes, Linear Support Vector Classifier, k-Nearest Neighbors, and Logistic Regression), as well as two text vectorizers (count and TF-IDF, for all but the LSTM models), in predicting if a news article is real or fake from its title and content. In this project, the LSTM neural network was the most accurate, followed by Random Forest, and there was no notable difference between the accuracies obtained from the use of either vectorizer.
