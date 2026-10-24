from pathlib import Path

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from pyspark.sql import SparkSession, functions as spark_functions
from sklearn.linear_model import LinearRegression


PROJECT_FOLDER = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_FOLDER / "data" / "synthetic_crime_data.csv"
LARGE_SAMPLE_CSV = PROJECT_FOLDER / "data" / "synthetic_crime_data_10000.csv"


st.set_page_config(
    page_title="Crime Data Analytics and Prediction System Using Cloud",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("CrimeDataAnalyticsDemo")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


@st.cache_data(show_spinner="Loading the synthetic CSV with PySpark...")
def load_crime_summaries(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spark = get_spark_session()
    crime_rows = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
        .withColumn("crime_date", spark_functions.to_date("crime_date"))
        .withColumn(
            "month",
            spark_functions.date_format(spark_functions.col("crime_date"), "yyyy-MM"),
        )
    )

    category_counts = (
        crime_rows.groupBy("category")
        .count()
        .orderBy(spark_functions.desc("count"))
        .toPandas()
    )
    monthly_counts = (
        crime_rows.groupBy("month")
        .count()
        .orderBy("month")
        .toPandas()
    )
    return category_counts, monthly_counts


def fill_missing_months(monthly_counts: pd.DataFrame) -> pd.DataFrame:
    monthly_counts = monthly_counts.copy()
    monthly_counts["month_start"] = pd.to_datetime(monthly_counts["month"] + "-01")
    all_months = pd.date_range(
        start=monthly_counts["month_start"].min(),
        end=monthly_counts["month_start"].max(),
        freq="MS",
    )
    monthly_counts = (
        monthly_counts.set_index("month_start")
        .reindex(all_months, fill_value=0)
        .rename_axis("month_start")
        .reset_index()
    )
    monthly_counts["month"] = monthly_counts["month_start"].dt.strftime("%Y-%m")
    return monthly_counts


def predict_next_month(monthly_counts: pd.DataFrame) -> tuple[str, int]:
    model = LinearRegression()
    month_numbers = [[number] for number in range(len(monthly_counts))]
    model.fit(month_numbers, monthly_counts["count"])

    next_month_number = [[len(monthly_counts)]]
    predicted_count = max(0, round(float(model.predict(next_month_number)[0])))
    next_month = pd.Period(monthly_counts["month"].iloc[-1], freq="M") + 1
    return str(next_month), predicted_count


st.title("Crime Data Analytics and Prediction System Using Cloud")

if not SAMPLE_CSV.exists():
    st.error(f"Sample dataset not found: {SAMPLE_CSV}")
    st.stop()

available_datasets = {"Small sample (25 records)": SAMPLE_CSV}
if LARGE_SAMPLE_CSV.exists():
    available_datasets["Large synthetic sample (10,000 records)"] = LARGE_SAMPLE_CSV

selected_dataset = st.selectbox("Dataset", list(available_datasets))
selected_csv = available_datasets[selected_dataset]
category_counts, monthly_counts = load_crime_summaries(str(selected_csv))
monthly_counts = fill_missing_months(monthly_counts)

if category_counts.empty or monthly_counts.empty:
    st.error("The CSV does not contain rows that can be analyzed.")
    st.stop()

total_reports = int(category_counts["count"].sum())
busiest_category = str(category_counts.iloc[0]["category"])
monthly_average = monthly_counts["count"].mean()

metric_columns = st.columns(4)
metric_columns[0].metric("Total synthetic reports", total_reports)
metric_columns[1].metric("Crime categories", len(category_counts))
metric_columns[2].metric("Most common category", busiest_category)
metric_columns[3].metric("Monthly average", f"{monthly_average:.1f}")

chart_columns = st.columns(2)
with chart_columns[0]:
    category_chart = px.bar(
        category_counts,
        x="category",
        y="count",
        color="category",
        text="count",
        title="Reports by crime category",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    category_chart.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Reports")
    st.plotly_chart(category_chart, use_container_width=True)

with chart_columns[1]:
    monthly_chart = px.line(
        monthly_counts,
        x="month",
        y="count",
        markers=True,
        title="Reports by month",
    )
    monthly_chart.update_layout(xaxis_title="Month", yaxis_title="Reports")
    st.plotly_chart(monthly_chart, use_container_width=True)

if len(monthly_counts) >= 2:
    forecast_month, forecast_count = predict_next_month(monthly_counts)
    st.subheader("Simple monthly prediction")
    st.metric(f"Estimated reports for {forecast_month}", forecast_count)
    st.caption(
        "A scikit-learn LinearRegression model fits a straight-line trend to the "
        "monthly counts, then estimates the next month. This is a learning demo, "
        "not a reliable forecast."
    )
else:
    st.info("At least two months of data are needed to demonstrate the prediction.")

with st.expander("View PySpark summary tables"):
    table_columns = st.columns(2)
    table_columns[0].caption("Reports by category")
    table_columns[0].dataframe(
        category_counts.rename(columns={"category": "Category", "count": "Reports"}),
        use_container_width=True,
        hide_index=True,
    )
    table_columns[1].caption("Reports by month")
    table_columns[1].dataframe(
        monthly_counts[["month", "count"]].rename(
            columns={"month": "Month", "count": "Reports"}
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.sidebar:
    st.header("Optional Amazon S3 upload")
    st.caption(
        "Archive a CSV in your S3 bucket. The dashboard uses the bundled synthetic "
        "sample and works without AWS credentials."
    )
    s3_bucket = st.text_input("S3 bucket name")
    uploaded_csv = st.file_uploader("Choose a CSV to upload", type=["csv"])
    default_key = (
        f"uploads/{Path(uploaded_csv.name).name}"
        if uploaded_csv is not None
        else "uploads/crime_data.csv"
    )
    s3_object_key = st.text_input("S3 object key", value=default_key)
    upload_clicked = st.button(
        "Upload CSV to S3",
        disabled=not s3_bucket.strip() or not s3_object_key.strip() or uploaded_csv is None,
    )

    if upload_clicked and uploaded_csv is not None:
        try:
            s3_client = boto3.client("s3")
            s3_client.put_object(
                Bucket=s3_bucket.strip(),
                Key=s3_object_key.strip(),
                Body=uploaded_csv.getvalue(),
            )
        except (NoCredentialsError, ClientError, BotoCoreError) as error:
            st.error(
                f"S3 upload failed. Check your AWS credentials, bucket, and "
                f"upload permissions. Details: {error}"
            )
        else:
            try:
                uploaded_object = s3_client.head_object(
                    Bucket=s3_bucket.strip(),
                    Key=s3_object_key.strip(),
                )
            except (NoCredentialsError, ClientError, BotoCoreError) as error:
                st.warning(
                    "The CSV was uploaded, but S3 could not verify its metadata. "
                    f"Check s3:GetObject permission. Details: {error}"
                )
            else:
                stored_size = uploaded_object.get("ContentLength", 0)
                st.success(
                    f"Upload verified: s3://{s3_bucket.strip()}/{s3_object_key.strip()} "
                    f"({stored_size:,} bytes)"
                )
