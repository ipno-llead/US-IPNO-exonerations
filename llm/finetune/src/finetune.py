# +
import openai
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading API key file: {e}")
        return None

def main(input_file, api_key_file):
    api_key = read_api_key(api_key_file)
    if api_key is None:
        logging.error("API key is missing. Cannot proceed.")
        return

    openai.api_key = api_key

    try:
        with open(input_file, "r") as file:
            logging.info("Uploading file for fine-tuning...")
            ids = openai.File.create(file=file, purpose="fine-tune")
    except Exception as e:
        logging.error(f"Error in file upload: {e}")
        return

    file_id = ids["id"]
    logging.info(f"File uploaded successfully. File ID: {file_id}")

    try:
        logging.info("Creating fine-tuning job...")
        create_job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo-0613")
    except Exception as e:
        logging.error(f"Error in creating fine-tuning job: {e}")
        return

    job_id = create_job["id"]
    logging.info(f"Fine-tuning job created. Job ID: {job_id}")

    try:
        logging.info("Retrieving fine-tuning job details...")
        job_details = openai.FineTuningJob.retrieve(job_id)
        logging.info(f"Job details: {job_details}")
    except Exception as e:
        logging.error(f"Error in retrieving job details: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune GPT model.")
    parser.add_argument("--input", required=True, help="Path to the input training data file.")
    parser.add_argument("--apikey", required=True, help="Path to the API key file.")
    args = parser.parse_args()
    main(args.input, args.apikey)

