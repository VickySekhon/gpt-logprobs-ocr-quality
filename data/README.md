## How to Obtain the Dataset

1) Navigate [here](https://orda.shef.ac.uk/articles/dataset/BLN600_A_Parallel_Corpus_of_Machine_Human_Transcribed_Nineteenth_Century_Newspaper_Texts/25439023)
2) Find and click the button that says `Download all`
3) Locate the downloaded ZIP file and extract it:

### Windows

1) Right-click the file in Downloads and click `Extract All...`, choosing where to extract.
2) Enter password `BLN600` when prompted.

### macOS/Linux

1) Open a terminal in your Downloads folder (or provide the full path to the zip).
2) Extract the archive:

```bash
unzip BLN600.zip -d bln600
```

3) When prompted, enter password `BLN600`.

### Rename Folders

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

The pipeline expects a different naming convention. Rename the folders as follows:
- `Ground Truth` -> `ground-truth`
- `Images` -> `images`
- `OCR Text` -> `ocr-text`

### Copy Folders to Project Directory

From the project root, run the appropriate command below.

#### Windows

```powershell
Copy-Item -Path "C:\path\to\bln600\*" -Destination ".\data\" -Recurse
```

#### macOS/Linux

```bash
cp -r /path/to/bln600/. ./data/
```

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