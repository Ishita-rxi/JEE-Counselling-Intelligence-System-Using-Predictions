from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
all_data = []

def scrape_table(year):
    print(f"\n🔍 Waiting for table for {year}...")

    table = None

    for _ in range(15):
        tables = driver.find_elements(By.TAG_NAME, "table")

        if tables:
            table = tables[0]
            rows = table.find_elements(By.TAG_NAME, "tr")

            if len(rows) > 1:
                break

        time.sleep(1)

    if not table:
        print(" Table not found. Try again.")
        return

    rows = table.find_elements(By.TAG_NAME, "tr")
    print(f" Rows found: {len(rows)}")

    for i, row in enumerate(rows):
        try:
            cols = row.find_elements(By.TAG_NAME, "td")
            data = [col.text.strip() for col in cols]

            if data:
                data.insert(0, year)
                all_data.append(data)

            if i % 500 == 0:
                print(f"Processed {i} rows...")

        except:
            continue


print("\n========== 2025 ==========")

driver.get("https://josaa.admissions.nic.in/applicant/SeatAllotmentResult/CurrentORCR.aspx")

time.sleep(5)

print(" Select filters in browser (Round, Institute, Branch, Category)")
print(" Click SUBMIT and WAIT until table is fully visible")

input(" Press ENTER after table is visible...")

scrape_table("2025")


print("\n Moving to ARCHIVE page...")

driver.get("https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx")

time.sleep(5)

print(" Archive page opened")

years = ["2024", "2023"]


for year in years:
    print(f"\n========== {year} ==========")
    print(f" Select YEAR = {year} in browser")
    print(" Then choose filters and click SUBMIT")
    print(" WAIT for table to load")

    input(" Press ENTER after table is visible...")

    scrape_table(year)



print("\n Saving data...")

with open("josaa_all_years.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Year",
        "Institute",
        "Program",
        "Quota",
        "Seat Type",
        "Gender",
        "Opening Rank",
        "Closing Rank"
    ])

    writer.writerows(all_data)

print(" Data saved as josaa_all_years.csv")

driver.quit()