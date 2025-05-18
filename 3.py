import pymongo
import pandas as pd
from datetime import datetime

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["expenseDB"]
classified_col = db["cleaned_classified"]
monthly_summary_col = db["monthlycharts"]

def batch_update_monthly_summary():
    # Step 1: Load all classified data
    data = list(classified_col.find({}, {"_id": 0, "Date": 1, "Amount": 1}))
    df = pd.DataFrame(data)
    
    # Step 2: Clean and convert
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Amount'])

    # Step 3: Filter out entries from the year 2023
    df = df[df['Date'].dt.year != 2023]
    
    # Step 4: Extract month
    df['month'] = df['Date'].dt.strftime('%B %Y')
    df['monthNumber'] = df['Date'].dt.month

    # Step 5: Aggregate spend
    monthly_agg = df.groupby(['month', 'monthNumber']).agg(
        value=('Amount', 'sum')
    ).reset_index()

    # Step 6: Save to MongoDB
    monthly_summary_col.delete_many({})
    monthly_summary_col.insert_many(monthly_agg.to_dict('records'))

    print(f"✅ Inserted {len(monthly_agg)} records to 'monthly_spending_summary'.")

if __name__ == "__main__":
    batch_update_monthly_summary()
