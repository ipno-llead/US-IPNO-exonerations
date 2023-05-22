# Wrongful Conviction Data Processing Pipeline

The pipeline aims to assist in the analysis of wrongful conviction case files by performing various tasks such as creating an index, identifying specific case files, generating thumbnails, and extracting structured information.

## Overview

The data processing pipeline is divided into several parts:

### Part 1: Creating CSV Index

In this part, a CSV index is created for the 100GB of case files. The index contains the following columns:
- Filepath: The path of the case file.
- Filename: The name of the case file.
- Filehash: The hash value of the file.
- Filesize: The size of the file.
- Unique Identifier: A unique identifier for each case file.
- Filetype: The type of the file.
- Case ID: The ID of the case.

### Part 2: Identifying Case Files

In this part, specific case files are identified based on their filenames. The identification is done by checking if the filename contains keywords such as "police report," "testimony," or "transcript." This step helps generate training data for the model.

### Part 3: Generating Thumbnails

This part involves generating thumbnail PNG images for each case file. The thumbnails are used to train an image classifier that can identify police reports, transcripts, and testimonies that couldn't be identified solely by filename.

### Part 4: Extracting Structured Information

In this final part, structured information is extracted from the identified police reports, transcripts, and testimonies. Techniques such as text extraction and the use of libraries like Spacy and Microsoft Form Recognizer are employed to extract relevant information such as police officers' names, ranks, and roles in the case files.
