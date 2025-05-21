import pymongo
import re
import time
from datetime import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# MongoDB Connection
connection_string = "mongodb://localhost:27017/"
client = pymongo.MongoClient(connection_string)

# Databases and Collections
db = client["expenseDB"]
raw_collection = db["raw_sms"]
processed_collection = db["transactions"]
categorized_collection = db["categorized_transactions"]
training_collection = db["training_data"]
manual_collection = db["expenses"]
combined_collection = db["categorycharts"]

# Initialize ML Components
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
label_encoder = LabelEncoder()
model = None

# Category Mapping
category_mapping = {
    'Essentials': 'Essentials',
    'Food & Dining': 'Food',
    'Shopping': 'Shopping',
    'Entertainment & Lifestyle': 'Lifestyle',
    'Savings & Transfers': 'Savings',
    'Travel & Transportation': 'Travel',
    'Subscriptions & Services': 'Subscription',
    'Miscellaneous': 'Misc.'
}

# ===========================
# ⚙️ Model Preparation & Utils
# ===========================

def initialize_training_data():
    global model, vectorizer, label_encoder
    try:
        with open('transaction_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('transaction_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('transaction_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Loaded existing model")
    except:
        default_data = pd.read_csv("Cleaned_Classified.csv")
        prepare_model(default_data)
        print("✅ Initialized new model with default data")

def prepare_model(training_data):
    global model, vectorizer, label_encoder
    training_data['cleaned_body'] = training_data['Body'].apply(clean_text)
    label_encoder.fit(training_data['Category'])
    y = label_encoder.transform(training_data['Category'])
    X = vectorizer.fit_transform(training_data['cleaned_body'])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    with open('transaction_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('transaction_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('transaction_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text)).lower()

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
        if transaction_type.lower() == "credited":
            return "Savings & Transfers"
        return category
    except Exception as e:
        print(f"⚠️ Classification error: {e}")
        return "Miscellaneous"

def initialize_processed_ids():
    processed_ids = set()
    # Track both SMS and manual entries
    for doc in processed_collection.find({}, {"_id": 1}):
        processed_ids.add(doc["_id"])
    for doc in categorized_collection.find({}, {"_id": 1}):
        processed_ids.add(doc["_id"])
    for doc in manual_collection.find({}, {"_id": 1}):
        processed_ids.add(doc["_id"])
    return processed_ids

def update_combined_transactions():
    """Update the combined transactions view with latest data"""
    combined_collection.delete_many({})
    category_totals = {}

    # Process SMS transactions
    for doc in categorized_collection.find({}):
        original_cat = doc.get("Category", "Miscellaneous")
        if original_cat in category_mapping:
            new_cat = category_mapping[original_cat]
            amt = float(doc.get("Amount", 0))
            category_totals[new_cat] = category_totals.get(new_cat, 0) + amt

    # Process manual entries
    for doc in manual_collection.find({}):
        original_cat = doc.get("expenseName", "Miscellaneous")
        if original_cat in category_mapping:
            new_cat = category_mapping[original_cat]
            amt = float(doc.get("expenseAmount", 0))
            category_totals[new_cat] = category_totals.get(new_cat, 0) + amt

    # Update combined collection
    for category, total in category_totals.items():
        combined_collection.update_one(
            {"_id": category},
            {"$set": {"category": category, "value": round(total, 2)}},
            upsert=True
        )
    print("✅ Updated combined transactions")

# ===========================
# 🔁 Real-Time Processing
# ===========================

processed_ids = initialize_processed_ids()
initialize_training_data()

transaction_categories = ["debited", "credited", "sent", "received", "payment successful", "amount sent"]
non_transactional = ["offer", "promotion", "hurry", "discount", "click here", "thanks for recharge"]

print("🔄 Real-time SMS + Manual Entry Processing Started")

# Track last counts for change detection
last_sms_count = raw_collection.count_documents({})
last_manual_count = manual_collection.count_documents({})

while True:
    try:
        # Check for new manual entries
        current_manual_count = manual_collection.count_documents({})
        if current_manual_count != last_manual_count:
            print("🆕 New manual entries detected")
            # Get all manual IDs to update processed_ids
            manual_ids = set(doc["_id"] for doc in manual_collection.find({}, {"_id": 1}))
            processed_ids.update(manual_ids)
            last_manual_count = current_manual_count
            
            # Update combined transactions
            update_combined_transactions()
            continue

        # Check for new SMS
        current_sms_count = raw_collection.count_documents({})
        if current_sms_count != last_sms_count:
            print("📨 New SMS detected")
            last_sms_count = current_sms_count
            new_sms_processed = False
            
            # Process new SMS
            for sms in raw_collection.find({"_id": {"$nin": list(processed_ids)}}):
                sms_id = sms["_id"]
                sms_body = sms.get("body", "")
                sms_address = sms.get("address", "Unknown")
                sms_date = sms.get("readable_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                if not any(cat in sms_body.lower() for cat in transaction_categories) or \
                   any(non_trans in sms_body.lower() for non_trans in non_transactional):
                    processed_ids.add(sms_id)
                    continue

                amount, recipient, transaction_type = parse_transaction_details(sms_body)
                amount_float = float(amount) if amount else 0
                category = classify_transaction(sms_body, transaction_type)

                # Update transactions collection
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

                # Update categorized transactions
                categorized_data = {
                    "_id": sms_id,
                    "Date": sms_date,
                    "Address": sms_address,
                    "Amount": amount_float,
                    "Recipient": recipient,
                    "Transaction Type": transaction_type,
                    "Body": sms_body,
                    "Category": category,
                    "Verified": category != "Miscellaneous"
                }
                categorized_collection.update_one(
                    {"_id": sms_id},
                    {"$setOnInsert": categorized_data},
                    upsert=True
                )
                processed_ids.add(sms_id)
                print(f"✅ Processed SMS: {transaction_type} {amount_float} to {recipient} → {category}")
                new_sms_processed = True

            if new_sms_processed:
                update_combined_transactions()

        time.sleep(5)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(10)