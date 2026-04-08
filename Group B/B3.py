# ============================================================
#   MapReduce: Find Hottest and Coolest Year
# ============================================================
# HOW TO RUN:
#   python B3.py
#
# This program simulates a MapReduce distributed environment
# using Python's pure map() and reduce() functions along with 
# collections for the shuffle/sort phase. 
# ============================================================

import csv
import os
from functools import reduce
from collections import defaultdict

# ── STEP 0: Create Mock Weather Data if it doesn't exist ─────
DATA_FILE = "weather_data.csv"

def create_sample_data():
    if not os.path.exists(DATA_FILE):
        print(f"[*] Creating sample weather data: {DATA_FILE}")
        data = [
            ("2020-05-12", 34.5), ("2020-12-01", 5.2),
            ("2021-06-15", 36.1), ("2021-11-20", 4.1),
            ("2022-05-30", 38.0), ("2022-12-25", 2.0),
            ("2023-04-10", 35.5), ("2023-11-11", 6.0),
            ("2019-07-07", 33.0), ("2019-12-30", 3.5),
        ]
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Temperature"])
            for row in data:
                writer.writerow(row)

# ── STEP 1: Mapper Function ──────────────────────────────────
# Input : A line of data (Date, Temperature)
# Output: Key-Value pair -> (Year, Temperature)
def mapper(record):
    date_str, temp_str = record
    if date_str == "Date": # Skip header
        return None
    year = date_str.split("-")[0]
    try:
        temp = float(temp_str)
        return (year, temp)
    except ValueError:
        return None


# ── STEP 2: Shuffle and Sort Phase ───────────────────────────
# Input : List of (Year, Temperature)
# Output: Dictionary -> { Year: [temp1, temp2, ...] }
def shuffle_and_sort(mapped_data):
    grouped_data = defaultdict(list)
    for kv in mapped_data:
        if kv is not None:
            year, temp = kv
            grouped_data[year].append(temp)
    return grouped_data


# ── STEP 3: Reducer Function ─────────────────────────────────
# Input : A tuple -> (Year, [temp1, temp2, temp3...])
# Output: Evaluated -> (Year, HighestTemp, LowestTemp)
def reducer(item):
    year, temps = item
    
    # Use python's reduce to find max and min
    max_temp = reduce(lambda a, b: a if a > b else b, temps)
    min_temp = reduce(lambda a, b: a if a < b else b, temps)
    
    return (year, max_temp, min_temp)


# ── MAIN EXECUTION ───────────────────────────────────────────
if __name__ == "__main__":
    create_sample_data()

    print("=" * 55)
    print("   MapReduce: Hottest & Coolest Year Finder")
    print("=" * 55)

    # 1. READ Data
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        raw_data = list(reader)

    # 2. MAP: apply mapper to each record
    print("[1] Running MAP phase...")
    mapped_data = list(map(mapper, raw_data))

    # 3. SHUFFLE & SORT: group temps by year
    print("[2] Running SHUFFLE & SORT phase...")
    shuffled_data = shuffle_and_sort(mapped_data)

    # 4. REDUCE: apply reducer to evaluate max and min for years
    print("[3] Running REDUCE phase...")
    reduced_data = list(map(reducer, shuffled_data.items()))

    # Calculate overall coolest and hottest years from the reduced data
    overall_hottest = max(reduced_data, key=lambda x: x[1])  # Compare by max_temp
    overall_coolest = min(reduced_data, key=lambda x: x[2])  # Compare by min_temp

    # ── OUTPUT RESULTS ──
    print("-" * 55)
    print("   Yearly Extremes Computed:")
    for res in sorted(reduced_data, key=lambda x: x[0]):
        print(f"     Year: {res[0]} | High: {res[1]:.1f}°C | Low: {res[2]:.1f}°C")
    
    print("-" * 55)
    print(f"🔥 HOTTEST YEAR: {overall_hottest[0]} (Max Temp: {overall_hottest[1]:.1f}°C)")
    print(f"❄️ COOLEST YEAR: {overall_coolest[0]} (Min Temp: {overall_coolest[2]:.1f}°C)")
    print("=" * 55)
