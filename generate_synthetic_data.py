import csv
import random
from datetime import date, timedelta
from pathlib import Path


OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "synthetic_crime_data_10000.csv"
RECORD_COUNT = 10_000
RANDOM_SEED = 42
CATEGORIES = ["Theft", "Burglary", "Vehicle Theft", "Fraud", "Assault"]
CATEGORY_WEIGHTS = [35, 22, 18, 15, 10]
DISTRICTS = ["North", "South", "East", "West", "Central"]
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)


def main() -> None:
    if OUTPUT_CSV.exists():
        raise SystemExit(
            f"Refusing to overwrite existing file: {OUTPUT_CSV}"
        )

    random_generator = random.Random(RANDOM_SEED)
    date_count = (END_DATE - START_DATE).days + 1
    field_names = ["report_id", "crime_date", "category", "district", "data_label"]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

        for record_number in range(1, RECORD_COUNT + 1):
            crime_date = START_DATE + timedelta(
                days=random_generator.randrange(date_count)
            )
            category = random_generator.choices(
                CATEGORIES, weights=CATEGORY_WEIGHTS, k=1
            )[0]
            district = random_generator.choice(DISTRICTS)
            writer.writerow(
                {
                    "report_id": f"SYN-LARGE-{record_number:05d}",
                    "crime_date": crime_date.isoformat(),
                    "category": category,
                    "district": district,
                    "data_label": "Synthetic sample data",
                }
            )

    print(f"Created {RECORD_COUNT:,} synthetic records at {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
