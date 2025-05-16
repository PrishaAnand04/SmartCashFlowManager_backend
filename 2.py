import pandas as pd
import pymongo
from datetime import datetime
import matplotlib.colors as mcolors

# MongoDB Connection
connection_string = "mongodb://localhost:27017/"
client = pymongo.MongoClient(connection_string)

# Databases and Collections
db = client["expenseDB"]
transactions_col = db["cleaned_classified"]
predictions_col = db["summaries"]

def load_data_to_mongodb(file_path):
    """Load CSV data into MongoDB transactions collection"""
    data = pd.read_csv(file_path)
    
    # Convert to datetime and handle missing values
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
    data = data.dropna(subset=['Date', 'Category'])
    
    # Convert to dictionary and insert
    records = data.to_dict('records')
    transactions_col.insert_many(records)
    print(f"Inserted {len(records)} transactions into MongoDB")

def generate_predictions():
    """Generate simplified monthly spending predictions and store in MongoDB"""
    all_data = list(transactions_col.find({}, {'_id': 0, 'Date': 1, 'Amount': 1, 'Category': 1}))
    df = pd.DataFrame(all_data)

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Date', 'Amount', 'Category'])

    df['Month'] = df['Date'].dt.to_period('M')

    current_month = pd.Period("2024-11", freq='M')
    previous_month = current_month - 1

    filtered_data = df[df['Month'] <= previous_month]

    monthly_stats = filtered_data.groupby(['Month', 'Category']).agg(
        Total_Spending=('Amount', 'sum'),
        Transaction_Count=('Amount', 'count')
    ).reset_index()

    category_stats = monthly_stats.groupby('Category').agg(
        Avg_Spending=('Total_Spending', 'mean'),
        Min_Spending=('Total_Spending', 'min'),
        Max_Spending=('Total_Spending', 'max'),
        Total_Transactions=('Transaction_Count', 'sum')
    ).reset_index()

    predictions = []
    color_palette = list(mcolors.TABLEAU_COLORS.values())

    for idx, row in category_stats.iterrows():
        avg = row['Avg_Spending']
        min_range = round(avg * 0.85, 2)
        max_range = round(avg * 1.15, 2)
        color = color_palette[idx % len(color_palette)]

        prediction = {
            'category': row['Category'],
            'value': round(avg, 2),
            'colorHex': color,
            'range': f"{min_range} - {max_range}"
        }
        predictions.append(prediction)

    # Clear previous predictions
    predictions_col.delete_many({})

    if predictions:
        predictions_col.insert_many(predictions)
        print(f"Generated and inserted {len(predictions)} simplified predictions.")
    return predictions

def get_latest_predictions():
    """Retrieve the latest predictions from MongoDB"""
    return list(predictions_col.find({}, {'_id': 1, 'category': 1, 'value': 1, 'colorHex': 1, 'range': 1}))

# Example usage
if __name__ == "__main__":
    # Step 1: Load data (run this once)
    # load_data_to_mongodb('Cleaned_Classified.csv')
    
    # Step 2: Generate predictions (run monthly)
    generate_predictions()
    
    # Step 3: Retrieve and display predictions
    latest_preds = get_latest_predictions()
    for pred in latest_preds:
        print(f"Category: {pred['category']}")
        print(f"Average Prediction: {pred['value']}")
        print(f"Expected Range: {pred['range']}")
        print(f"Color: {pred['colorHex']}")
        print("---")
