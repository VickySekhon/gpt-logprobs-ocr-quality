### How to Obtain the Dataset

1) Navigate [here](https://orda.shef.ac.uk/articles/dataset/BLN600_A_Parallel_Corpus_of_Machine_Human_Transcribed_Nineteenth_Century_Newspaper_Texts/25439023)
2) Find and click the button that says `Download all`
3) Find the downloaded zip file on your machine and extract it by doing the following:

#### Windows 

1) Right click the file in Downloads and click `Extract All...`, choosing where to extract.
2) Enter password `BLN600` when prompted.

#### MacOS/Linux

1) Open a terminal in your Downloads folder (or provide the full path to the zip).
2) Extract with password:
`unzip BLN600.zip -d bln600`
3) Enter password `BLN600` when prompted.

#### Rename Folders

The extracted ZIP archive contents will look like follows:
```
parent_dir/
     Ground Truth/
     Images/
     OCR Text/
     readme.md
     LICENSE
     metadata.json
```

The pipeline expects a separate naming convention. Please name folders manually by following the convention below:
- `Ground Truth` -> `ground-truth`
- `Images` -> `images`
- `OCR Text` -> `ocr-text`

#### Copy Folders to Project Directory

Ensure you are at the root of the project directory then follow the commands below.

#### Windows

`Copy-Item -Path "C:\path\to\bln600\*" -Destination ".\data\" -Recurse`

#### MacOS/Linux

`cp -r /path/to/bln600/. ./data/`

**Note**: the following files are not required to run the pipeline and can be safely deleted:

- `readme.md`
- `LICENSE`
- `metadata.json`

Upon successfully following the steps, `gpt-logprobs-ocr-quality/data/` should look like:
```
data/
     ground-truth/
     images/
     ocr-text/
```