# Wrongful Conviction Data Processing Pipeline

The pipeline aims to assist in the analysis of wrongful conviction case files by performing various tasks such as creating an index, identifying specific case files, generating thumbnails, and extracting structured information.

## Overview

The data processing pipeline is divided into several parts:

### Part 1: Creating CSV Index

In this part, a CSV index is created for the 100GB of case files. The index contains the following columns:

Filepath: The path of the case file.
Filename: The name of the case file.
Filehash: The hash value of the file.
Filesize: The size of the file.
Unique Identifier: A unique identifier for each case file.
Filetype: The type of the file.
Case ID: The ID of the case.

### Part 2: Identifying Case Files

In this part, specific case files are identified based on their filenames. The identification is done by checking if the filename contains keywords such as "police report," "testimony," or "transcript." This step helps generate training data for the model.

### Part 3: Generating Thumbnails

This part involves generating thumbnail PNG images for each case file. The thumbnails are used to train an image classifier that can identify police reports, transcripts, and testimonies that couldn't be identified solely by filename.

### Part 4: Extracting Structured Information

In this part, structured information is extracted from the identified police reports, transcripts, and testimonies. Techniques such as text extraction and the use of libraries like ChatGPT, Regex, Spacy and Microsoft Form Recognizer are employed to extract relevant information such as police officers' names, ranks, and roles in the case files.

### Part 5: Cross-Referencing Officer Names with Louisiana Law Enforcement Accountability Database

In this part, the data extracted from the case files, specifically the officer names and roles, will be cross-referenced with the Louisiana Law Enforcement Accountability Database. This database contains comprehensive data on over 60,000 police officers and 40,000 complaints of misconduct. By cross-referencing officers named in wrongful conviction case files, we aim to identify their partners, supervisors, and trainees who may have also engaged in behavior related to wrongful convictions.

Furthermore, with the names of all officers involved in wrongful conviction case files, we can request records that will allow us to know the identities of all the individuals they have ever arrested, possibly wrongly. This information will be cross-referenced with our internal database of jailed individuals who have claimed to be wrongfully convicted. By analyzing these cross-referenced records, we hope to uncover potential patterns and further support the identification of wrongful convictions.

To ensure the privacy and security of the data, appropriate measures will be taken to handle and store the sensitive information in compliance with legal and ethical standards.

### Part 6: Analysis and Reporting

After completing the data processing pipeline, the extracted information and cross-referenced records will be further analyzed to identify potential cases of wrongful conviction. Statistical analysis, data visualization, and machine learning techniques may be applied to uncover patterns, trends, and correlations that can contribute to the identification of systemic issues within the criminal justice system. The findings will be reported and shared with relevant stakeholders, advocacy groups, and legal experts to support efforts in rectifying wrongful convictions and improving the overall integrity of the justice system.

It is important to note that this pipeline is an iterative process, and feedback from experts and stakeholders will be taken into account to continuously improve the accuracy, efficiency, and ethical considerations of the pipeline's components.