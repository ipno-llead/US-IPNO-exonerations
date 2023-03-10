
- [Mathematics for Machine Learning](https://mml-book.github.io/)
- Tulane University supercomputer cluster

- RVL-CDiP: 

# Timeline

## Get files onto dropbox

- March 1

## Getting permission to look at the documents

Need data sharing agreement

- March 1

## Indexing

by March 7 or 8: we have an index that has at least:

- hash
- fileid
- case identifier and related case info
- file type (pdf or word or jpg)
- number of pages
- name of defendant/case

## Page classification

Steps: 

- Split each document into one row per page. 
- Filter for transcripts and police reports (we want 100ish examples of each document type). 
- Leverage the RVL-CDIP dataset to assist with thumb nail image classification (maybe someone has already created a model).
- Create training data based on string matching. 

We think this will take one month

done by early April


## text

end of april

- need to find out if we can use cloud service

## mention extraction from transcripts

- Use named entity recognition to extract data from transcripts.  


## mention extraction from police reports

- Use Microsoft Form Recognizer to extract key value pairs from forms.


## entity de-duplication

- We want to extract name, rank, badge number, agency and role.
- Any given classified page may have 0 or move officers mentioned. Each mention will reference all or some of the data referenced above. 
- With this index, we can de-duplicate again by looking at distinct mentions that reference the same person. From here we can assign a UID to an the entity (person) who can be associated with one or more mentions (events).

Sept-Oct

## November

We're done


To do: tarak (immediately)

- set up user accounts on eleanor for john and ayyub
- set up snap repo
