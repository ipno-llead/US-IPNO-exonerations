import pandas as pd
import os
from docx2pdf import convert

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to convert doc/docx files to PDF
def convert_docs_to_pdf(df):
    # Create a new column for PDF filepaths
    df = df[df.filetype(isin(["doc", "docx"])
    df['pdf_filepath'] = ''

    # Function to convert a doc/docx file to PDF
    def convert_to_pdf(file_path):
        pdf_path = file_path.replace('.docx', '.pdf').replace('.doc', '.pdf')
        convert(file_path, pdf_path)
        return pdf_path

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        file_path = row['filepath']

        # Check if the file exists
        if os.path.isfile(file_path):
            # Convert the file to PDF
            pdf_path = convert_to_pdf(file_path)

            # Update the PDF filepath in the DataFrame
            df.at[index, 'pdf_filepath'] = pdf_path
        else:
            print(f"File not found: {file_path}")

    return df

def change_dir(df):
    df.loc[:, "filepath"] = df.filepath.str.replace(r"^(.+)", r"../\1", regex=True)
    return df

if __name__ == "__main__":
    csv_path = args.input
    df = pd.read_csv(csv_path, sep="|")
    df = df.pipe(change_dir)
    df.to_csv(args.output)
    

