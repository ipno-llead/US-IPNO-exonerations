# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     TS
# Maintainers: TS
# Copyright:   2023, HRDAG, GPL v2 or later
# =========================================

import os
import argparse
import io
import json
import logging

from PyPDF2 import PdfFileReader, PdfFileWriter

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--creds")
    return parser.parse_args()


def getcreds(args):
    with open(args.creds, 'r') as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()


def subset_pdf(spec):
    pdf = PdfFileReader(spec.filename)
    pagenos = spec.pageno.split(' ')
    out_stream = io.BytesIO()
    output = PdfFileWriter()

    # spec page numbers are 1-indexed, but getPage expects zero-index
    for pn in pagenos:
        page = int(pn) - 1
        output.addPage(pdf.getPage(page))

    output.write(out_stream)
    # Move the stream position to the beginning,
    # making it easier for other code to read
    out_stream.seek(0)
    return out_stream

class DocClient:

    """client for form recognizer"""

    def __init__(self, endpoint, key):
        self.client = DocumentAnalysisClient(endpoint=endpoint,
                credential = AzureKeyCredential(key))

    def close(self):
        self.client.close()

    def pdf2json(self, pdf):
        pdf.seek(0)
        poller = self.client.begin_analyze_document(
                "prebuilt-document", document=pdf.read())
        result = poller.result()
        return result

    def process(self, spec):
        outstring = os.path.join('output/json', '{}.json'.format(spec.outname))
        outpath = os.path.abspath(outstring)
        if os.path.exists(outpath):
            logging.info(f"skipping {outpath}, file already exists")
            return outpath
        docpath = os.path.abspath(spec.filename)
        pdf = subset_pdf(spec)
        logging.info(f"sending document {spec.outname}")
        js = self.pdf2json(pdf)
        logging.info(f"writing to {outpath}")
        with open(outpath, "w") as f:
            json.dump(js.to_dict(), f)
        return outpath


if __name__ == '__main__':
    logger = logging.getLogger()
    azurelogger = logging.getLogger('azure')
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    args = getargs()
    endpoint, key = getcreds(args)
    client = DocClient(endpoint, key)

    driver = pd.read_parquet(args.input)
    logging.info(f"starting to process {len(driver)} files")
    for spec in driver.itertuples():
        client.process(spec)
    client.close()


# done.
