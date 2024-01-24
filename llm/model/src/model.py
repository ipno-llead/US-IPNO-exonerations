import os
import pandas as pd
import logging
import re
import argparse

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

iteration_times = 6
max_retries = 10

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Environment variables loaded.")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
TEMPERATURE = 1
k = 20


def clean_name(officer_name):
    return re.sub(
        r"(Detective|[Dd]et\.?:?\,?|[Ll]t\.?:?\,?|[Ss]gt\.?|Officer|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator|^-|\d{1}\)|\w{1}\.)\.?\s+",
        "",
        officer_name,
    )


def extract_officer_data(text):
    officer_data = []
    normalized_text = re.sub(r"\s*-\s*", "", text)
    officer_sections = re.split(r"\n(?=Officer Name:)", normalized_text)

    for section in officer_sections:
        if not section.strip():
            continue
        officer_dict = {}
        name_match = re.search(
            r"Officer Name:\s*(.*?)\s*Officer Context:", section, re.DOTALL
        )
        context_match = re.search(
            r"Officer Context:\s*(.*?)\s*Officer Role:", section, re.DOTALL
        )
        role_match = re.search(r"Officer Role:\s*(.*)", section, re.DOTALL)
        if name_match and name_match.group(1):
            officer_dict["Officer Name"] = clean_name(name_match.group(1).strip())
        if context_match and context_match.group(1):
            officer_dict["Officer Context"] = context_match.group(1).strip()
        if role_match and role_match.group(1):
            officer_dict["Officer Role"] = role_match.group(1).strip()

        if officer_dict:
            officer_data.append(officer_dict)
    return officer_data


def generate_hypothetical_embeddings():
    llm = OpenAI(openai_api_key="")
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings(openai_api_key="")

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings


def sort_retrived_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs


PROMPT_TEMPLATE_HYDE = PromptTemplate(
    input_variables=["question"],
    template="""
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, coroners, investigators, patrolman, and criminalists, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

    Question: {question}

    Roles and Responses:""",
)

PROMPT_TEMPLATE_MODEL = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of a law enforcement personnel. The individual's name must be prefixed with one of the following titles to be in law enforcement: 
       Detective, Sergeant, Lieutenant, Captain, Deputy, Officer, Patrol Officer, Criminalist, Technician, Coroner, or Dr. 
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, please make this clear in your response.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst
       Please prefix this information with "Officer Role: "
       For example, "Officer Role: Lead Detective"

    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith 
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: 
    Officer Context:
    Officer Role:    
    
    Officer Name: 
    Officer Context:
    Officer Role:   
""",
)


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def preprocess_document(file_path, embeddings):
    logger.info(f"Processing Word document: {file_path}")

    loader = JSONLoader(
        file_path,
        jq_schema=".messages[]",
        content_key="page_content",
        metadata_func=metadata_func,
    )
    text = loader.load()
    logger.info(f"Text loaded from JSON object: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, temperature, k):
    logger.info("Performing query...")

    doc_list = db.similarity_search_with_score(query, k=k)

    docs = sort_retrived_documents(doc_list)

    docs_page_content = " ".join([d[0].page_content for d in docs])

    # Initialize an empty list to store page numbers
    page_numbers = []

    # Loop through docs and extract page numbers, appending them to the list
    for doc in docs:
        page_number = doc[0].metadata.get("page_number")
        if page_number is not None:
            page_numbers.append(page_number)

    ft_model = "ft:gpt-3.5-turbo-0613:personal::8jvQ6VA6"

    llm = ChatOpenAI(
        model_name=ft_model,
        api_key="",
    )

    prompt = PROMPT_TEMPLATE_MODEL

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(
        question=query, docs=docs_page_content, temperature=temperature
    )

    return response, page_numbers


SINGLE_QUERY = [
    "Identify each individual in the transcript, by name, who are directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel. Provide the context of their mention, focusing on key events, significant decisions or actions they made, interactions with other individuals, roles or responsibilities they held, noteworthy outcomes or results they achieved, and any significant incidents or episodes they were involved in, if available."
]

MULTIPLE_QUERIES = [
    "Identify individuals, by name, with the specific titles of officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel in the transcript. Specifically, provide the context of their mention related to key events in the case, if available.",
    "List individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel mentioned in the transcript. Provide the context of their mention in terms of any significant decisions they made or actions they took.",
    "Locate individuals, by name, directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Explain the context of their mention in relation to their interactions with other individuals in the case.",
    "Highlight individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Describe the context of their mention, specifically noting any roles or responsibilities they held in the case.",
    "Outline individuals, by name, directly identified as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Specify the context of their mention in terms of any noteworthy outcomes or results they achieved.",
    "Pinpoint individuals, by name, directly labeled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Provide the context of their mention, particularly emphasizing any significant incidents or episodes they were involved in.",
]


# +
def process_files_in_directory(
    input_path, output_path, embeddings, multiple_queries_mode=False
):
    queries = (
        MULTIPLE_QUERIES
        if multiple_queries_mode
        else [SINGLE_QUERY[0]] * iteration_times
    )
    queries_label = "_multiple_queries" if multiple_queries_mode else "_single_query"

    for file_name in os.listdir(input_path):
        if file_name.endswith(".json"):
            csv_output_path = os.path.join(
                output_path, f"{file_name}{queries_label}.csv"
            )
            if os.path.exists(csv_output_path):
                logger.info(f"CSV output for {file_name} already exists. Skipping...")
                continue

            file_path = os.path.join(input_path, file_name)
            
            try:
                output_data = []
                
                try:
                    db = preprocess_document(file_path, embeddings)
                except Exception as e:
                    logger.error(f"Error preprocessing document {file_name}: {e}")
                    continue  # Skip to the next file
    
                for idx, query in enumerate(queries, start=1):
                    retries = 0
                    while retries < max_retries:
                        try:
                            officer_data_string, page_numbers = get_response_from_query(
                                db, query, TEMPERATURE, k
                            )
                            break
                        except ValueError as e:
                            if "Azure has not provided the response" in str(e):
                                retries += 1
                                logger.warn(
                                    f"Retry {retries} for query {query} due to Azure content filter error."
                                )
                            else:
                                raise

                    if retries == max_retries:
                        logger.error(f"Max retries reached for query {query}. Skipping...")
                        continue

                    officer_data = extract_officer_data(officer_data_string)

                    for item in officer_data:
                        item["page_number"] = page_numbers
                        item["fn"] = file_name
                        item["Query"] = query
                        item["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
                        item["Prompt Template for Model"] = PROMPT_TEMPLATE_MODEL
                        item["Chunk Size"] = CHUNK_SIZE
                        item["Chunk Overlap"] = CHUNK_OVERLAP
                        item["Temperature"] = TEMPERATURE
                        item["k"] = k
                        item["hyde"] = "1"
                        item["iteration"] = idx
                        item["num_of_queries"] = "6" if multiple_queries_mode else "1"
                        item["model"] = "gpt-3.5-turbo-1603-finetuned-300-labels"
                    output_data.extend(officer_data)

                output_df = pd.DataFrame(output_data)
                output_df.to_csv(csv_output_path, index=False)
            
            except Exception as e:
                logger.error(f"An error occurred while processing file {file_name}: {e}")
                continue 
            
def concatenate_csvs(input_directory, index_file_name):
    all_dataframes = []
    logger.info(f"Attempting to concatenate CSVs in directory: {input_directory}")

    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                logger.info(f"Adding CSV file to concatenation: {file_path}")
                df = pd.read_csv(file_path)
                all_dataframes.append(df)

    if not all_dataframes:
        logger.error("No CSV files found for concatenation.")
        return

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    output_file = os.path.join(input_directory, index_file_name)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined CSV created at: {output_file}")
    

def process_query(input_json_path, output_csv_path):
    embeddings = generate_hypothetical_embeddings()
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    logger.info(f"Processing file: {input_json_path}")
    process_files_in_directory(
        os.path.dirname(input_json_path),
        os.path.dirname(output_csv_path),
        embeddings
    )
    logger.info(f"Output CSV generated at: {output_csv_path}")
    
    return output_csv_path  

def main(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    csv_output_paths = []

    for index, row in df.iterrows():
        input_json_path = os.path.join('../../ocr/', row['json_filepath'])
        output_csv_path = row['json_filepath'].replace('.json', '.csv')
        output_csv_full_path = os.path.join(output_dir, output_csv_path)
        
        csv_output_path = process_query(input_json_path, output_csv_full_path)
        csv_output_paths.append(csv_output_path)

    concatenate_csvs(output_dir, "combined_output_reports.csv")
    logger.info("All CSV files have been concatenated into a single file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transcripts and reports based on a CSV list.")
    parser.add_argument("--csv_path", required=True, help="CSV file path with the list of files to process")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed CSV files")
    
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
