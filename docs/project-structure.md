## index-files

- hash each file
- grab metadata: file type, file size, ???
- there is additional file-type specific extractable info that might be useful
  here, for example number of pages for pdfs, dimensions for images, and
  wordcount for word/rtf docs.

../
├── docs
│   └── frozen
└── index
│   ├── input
│   └── src
└── classify-pages
│   ├── import
│   ├── thumbnail
│   ├── traindata
│   ├── train
│   ├── predict
│   └── export
└── ...

## classify-pages


- import (filter for filetypes)
- thumbnail (will need to handle both image and pdf files)
