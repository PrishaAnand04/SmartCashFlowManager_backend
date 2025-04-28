import pymongo
import re
import time
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# MongoDB Connection
connection_string = "mongodb://localhost:27017/"
client = pymongo.MongoClient(connection_string)

# Databases and Collections
raw_db = client["expenseDB"]
raw_collection = raw_db["raw_sms"]

processed_db = client["expenseDB"]
processed_collection = processed_db["transactions"]
categorized_collection = processed_db["categorized_transactions"]
training_collection = processed_db["training_data"]

# Initialize ML components
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
label_encoder = LabelEncoder()
model = None

# Load or initialize training dataset
def initialize_training_data():
    global model, vectorizer, label_encoder
    
    try:
        # Try to load existing model
        with open('transaction_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('transaction_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('transaction_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Loaded existing trained model")
    except:
        # Initialize with default dataset
        default_data = pd.read_csv("Cleaned_Classified.csv")
        prepare_model(default_data)
        print("✅ Initialized with default training dataset")

def prepare_model(training_data):
    global model, vectorizer, label_encoder
    
    # Clean text
    training_data['cleaned_body'] = training_data['Body'].apply(clean_text)
    
    # Encode labels
    label_encoder.fit(training_data['Category'])
    y = label_encoder.transform(training_data['Category'])
    
    # Vectorize text
    X = vectorizer.fit_transform(training_data['cleaned_body'])
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save components
    with open('transaction_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('transaction_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('transaction_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def parse_transaction_details(body):
    money_pattern = r'(?i)(INR|Rs\.?|₹)\s?(\d+[,.]?\d*)'
    recipient_pattern = r'to\s([A-Za-z\s]+)'

    money_match = re.search(money_pattern, body)
    amount = money_match.group(2).replace(',', '') if money_match else '0'

    recipient_match = re.search(recipient_pattern, body, re.IGNORECASE)
    recipient = recipient_match.group(1).strip() if recipient_match else 'N/A'

    if "credited" in body.lower() or "received" in body.lower():
        transaction_type = "credited"
    elif "debited" in body.lower() or "sent" in body.lower():
        transaction_type = "debited"
    else:
        transaction_type = "N/A"

    return amount, recipient, transaction_type

def classify_transaction(body, transaction_type):
    if model is None or transaction_type == "N/A":
        return "Miscellaneous"
    
    try:
        cleaned_text = clean_text(body)
        text_tfidf = vectorizer.transform([cleaned_text])
        predicted = model.predict(text_tfidf)
        category = label_encoder.inverse_transform(predicted)[0]
        
        # Special handling for credited transactions
        if transaction_type.lower() == "credited":
            return "Savings & Transfers"
            
        return category
    except Exception as e:
        print(f"⚠️ Classification error: {e}")
        return "Miscellaneous"

def update_training_data(sms_body, correct_category):
    """Add new training example and retrain model"""
    try:
        # Store in training collection
        training_collection.insert_one({
            "body": sms_body,
            "category": correct_category,
            "timestamp": datetime.now()
        })
        
        # Prepare new training data
        training_data = list(training_collection.find({}, {"_id": 0, "body": 1, "category": 1}))
        df = pd.DataFrame(training_data)
        df.columns = ['Body', 'Category']
        
        # Add default data if available
        try:
            default_data = pd.read_csv("Cleaned_Classified.csv")
            df = pd.concat([df, default_data], ignore_index=True)
        except:
            pass
        
        # Retrain model
        prepare_model(df)
        print(f"✅ Model updated with new training example: {correct_category}")
    except Exception as e:
        print(f"⚠️ Failed to update training data: {e}")

def initialize_processed_ids():
    processed_ids = set()
    for doc in processed_collection.find({}, {"_id": 1}):
        processed_ids.add(doc["_id"])
    for doc in categorized_collection.find({}, {"_id": 1}):
        processed_ids.add(doc["_id"])
    return processed_ids

# Initialize system
processed_ids = initialize_processed_ids()
initialize_training_data()

# Categories for filtering
transaction_categories = ["debited", "credited", "sent", "received", "payment successful", "amount sent"]
non_transactional = ["offer", "promotion", "hurry", "discount", "click here", "thanks for recharge"]

print("🔄 Real-time SMS processing and classification system ready")

while True:
    try:
        new_sms_docs = raw_collection.find({"_id": {"$nin": list(processed_ids)}})
        
        for sms in new_sms_docs:
            sms_id = sms["_id"]
            sms_body = sms.get("body", "")
            sms_address = sms.get("address", "Unknown")
            sms_date = sms.get("readable_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Skip non-transactional messages
            if not any(cat in sms_body.lower() for cat in transaction_categories) or \
               any(non_trans in sms_body.lower() for non_trans in non_transactional):
                processed_ids.add(sms_id)
                continue

            # Process transaction
            amount, recipient, transaction_type = parse_transaction_details(sms_body)
            amount_float = float(amount) if amount not in [None, "N/A"] else 0
            category = classify_transaction(sms_body, transaction_type)

            # Check if we need human verification
            if category == "Miscellaneous" and amount_float > 2000:  # Threshold for verification
                print(f"\n⚠️ High-value unclassified transaction detected:")
                print(f"Amount: {amount_float}")
                print(f"Body: {sms_body[:200]}...")
                new_category = input("Please enter correct category (or press Enter to keep as Miscellaneous): ")
                if new_category:
                    category = new_category
                    update_training_data(sms_body, category)

            # Store in databases
            transaction_data = {
                "_id": sms_id,
                "Date": sms_date,
                "Address": sms_address,
                "Amount": amount_float,
                "Recipient": recipient,
                "Transaction Type": transaction_type,
                "Body": sms_body
            }
            processed_collection.update_one(
                {"_id": sms_id},
                {"$setOnInsert": transaction_data},
                upsert=True
            )

            categorized_data = {
                "_id": sms_id,
                "Date": sms_date,
                "Address": sms_address,
                "Amount": amount_float,
                "Recipient": recipient,
                "Transaction Type": transaction_type,
                "Body": sms_body,
                "Category": category,
                "Verified": category != "Miscellaneous"  # Mark if verified or default
            }
            categorized_collection.update_one(
                {"_id": sms_id},
                {"$setOnInsert": categorized_data},
                upsert=True
            )

            print(f"✅ Processed: {transaction_type} {amount} to {recipient} ({category})")
            processed_ids.add(sms_id)
        
        time.sleep(5)
    
    except Exception as e:
        print(f"⚠️ System error: {e}")
        time.sleep(10)