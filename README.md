# Crime Data Analytics and Prediction System Using Cloud

A beginner-friendly college project that analyzes synthetic CSV data with Apache PySpark in local mode, displays the summaries in Streamlit, and demonstrates a simple monthly count estimate with scikit-learn LinearRegression.

## Important data note

`data/synthetic_crime_data.csv` contains 25 invented records and is clearly labeled **Synthetic sample data**. The optional larger dataset contains 10,000 generated records with the same columns. Neither dataset contains real personal data. The trend-based estimate is only a classroom demonstration; it is not an accurate or reliable real-world crime forecast.

## Requirements

- Python 3.9 or newer
- Java 11 or 17 installed and available on `PATH` for PySpark
- The Python packages listed in `requirements.txt`

## Setup on Windows PowerShell

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run the dashboard

To create the larger synthetic dataset, run this once from the project folder:

```powershell
python generate_synthetic_data.py
```

The script creates `data/synthetic_crime_data_10000.csv` with a fixed random seed, so the same 10,000 records can be used for a repeatable demonstration. It refuses to overwrite an existing output file. After generation, the dashboard's **Dataset** selector offers both the 25-record sample and the 10,000-record sample. If the larger CSV has not been generated, the dashboard continues to use the original sample.

With the virtual environment active, start the dashboard:

```powershell
streamlit run app.py
```

The app reads the selected CSV with PySpark using `local[*]`, groups reports by crime category and month, converts only the aggregated summaries to Pandas, and plots them with Plotly. The dashboard's record count, category chart, and monthly trend are the main analysis. LinearRegression provides a supporting straight-line trend estimate for the next month.

## Optional Amazon S3 upload

The local dashboard and charts do not require AWS. To use the optional sidebar upload, provide an S3 bucket name, select a CSV, and click **Upload CSV to S3**. Boto3 uses its standard AWS credential chain, such as environment variables, a configured AWS profile, or an assigned IAM role. The identity needs `s3:PutObject` to upload and `s3:GetObject` to verify the chosen object. After upload, the app calls S3 `head_object` and displays the stored object path and byte count. A successful `head_object` confirms that S3 can return the object's metadata; it does not download or compare the stored contents. Do not put AWS credentials in this project or commit them to source control.

The upload archives the selected CSV in S3; it does not replace the dataset used by the dashboard. If AWS is not configured, simply leave the S3 controls unused.
