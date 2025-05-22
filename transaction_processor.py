# transaction_processor.py
import pymongo
import re
import time
from datetime import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

class TransactionProcessor:
    def __init__(self, db_client):
        self.db = db_client["expenseDB"]
        self.initialize_collections()
        self.initialize_ml_components()
        self.processed_ids = self.initialize_processed_ids()
        
    def initialize_collections(self):
        self.raw_collection = self.db["raw_sms"]
        self.processed_collection = self.db["transactions"]
        self.categorized_collection = self.db["categorized_transactions"]
        self.manual_collection = self.db["expenses"]
        self.combined_collection = self.db["categorycharts"]
        
    def initialize_ml_components(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.model = None
        self.initialize_model()
        
        self.category_mapping = {
            'Essentials': 'Essentials',
            'Food & Dining': 'Food',
            'Shopping': 'Shopping',
            'Entertainment & Lifestyle': 'Lifestyle',
            'Savings & Transfers': 'Savings',
            'Travel & Transportation': 'Travel',
            'Subscriptions & Services': 'Subscription',
            'Miscellaneous': 'Misc.'
        }
        
        self.transaction_categories = ["debited", "credited", "sent", "received", 
                                     "payment successful", "amount sent"]
        self.non_transactional = ["offer", "promotion", "hurry", "discount", 
                                 "click here", "thanks for recharge"]
    
    def initialize_model(self):
        try:
            with open('transaction_classifier.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('transaction_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('transaction_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
        except:
            df = pd.read_csv("Cleaned_Classified.csv")
            self.train_model(df)
    
    def train_model(self, training_data):
        training_data['cleaned_body'] = training_data['Body'].apply(self.clean_text)
        y = self.label_encoder.fit_transform(training_data['Category'])
        X = self.vectorizer.fit_transform(training_data['cleaned_body'])
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        
        with open('transaction_classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('transaction_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('transaction_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def clean_text(self, text):
        return re.sub(r'[^\w\s]', '', str(text)).lower()
    
    def initialize_processed_ids(self):
        processed_ids = set()
        for doc in self.processed_collection.find({}, {"_id": 1}):
            processed_ids.add(doc["_id"])
        for doc in self.categorized_collection.find({}, {"_id": 1}):
            processed_ids.add(doc["_id"])
        for doc in self.manual_collection.find({}, {"_id": 1}):
            processed_ids.add(doc["_id"])
        return processed_ids
    
    def process_transactions(self):
        last_sms_count = self.raw_collection.count_documents({})
        last_manual_count = self.manual_collection.count_documents({})
        
        while True:
            try:
                current_manual_count = self.manual_collection.count_documents({})
                if current_manual_count != last_manual_count:
                    self.process_manual_entries()
                    last_manual_count = current_manual_count
                
                current_sms_count = self.raw_collection.count_documents({})
                if current_sms_count != last_sms_count:
                    self.process_sms_messages()
                    last_sms_count = current_sms_count
                
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                time.sleep(10)
    
    def process_manual_entries(self):
        print("🆕 New manual entries detected")
        manual_ids = {doc["_id"] for doc in self.manual_collection.find({}, {"_id": 1})}
        self.processed_ids.update(manual_ids)
        self.update_combined_transactions()
    
    def process_sms_messages(self):
        print("📨 New SMS detected")
        new_sms_processed = False
        
        for sms in self.raw_collection.find({"_id": {"$nin": list(self.processed_ids)}}):
            if self.process_single_sms(sms):
                new_sms_processed = True
                self.processed_ids.add(sms["_id"])
        
        if new_sms_processed:
            self.update_combined_transactions()
    
    def process_single_sms(self, sms):
        sms_body = sms.get("body", "")
        
        if (not any(cat in sms_body.lower() for cat in self.transaction_categories) or \
           any(non_trans in sms_body.lower() for non_trans in self.non_transactional)):
            return False
        
        amount, recipient, transaction_type = self.parse_transaction_details(sms_body)
        amount_float = float(amount) if amount else 0
        category = self.classify_transaction(sms_body, transaction_type)
        
        self.save_transaction(sms, amount_float, recipient, transaction_type, category, sms_body)
        return True
    
    def parse_transaction_details(self, body):
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
    
    def classify_transaction(self, body, transaction_type):
        if self.model is None or transaction_type == "N/A":
            return "Miscellaneous"
        try:
            cleaned_text = self.clean_text(body)
            text_tfidf = self.vectorizer.transform([cleaned_text])
            predicted = self.model.predict(text_tfidf)
            category = self.label_encoder.inverse_transform(predicted)[0]
            return "Savings & Transfers" if transaction_type.lower() == "credited" else category
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return "Miscellaneous"
    
    def save_transaction(self, sms, amount, recipient, transaction_type, category, body):
        transaction_data = {
            "_id": sms["_id"],
            "Date": sms.get("readable_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "Address": sms.get("address", "Unknown"),
            "Amount": amount,
            "Recipient": recipient,
            "Transaction Type": transaction_type,
            "Body": body
        }
        
        self.processed_collection.update_one(
            {"_id": sms["_id"]},
            {"$setOnInsert": transaction_data},
            upsert=True
        )
        
        categorized_data = {
            **transaction_data,
            "Category": category,
            "Verified": category != "Miscellaneous"
        }
        
        self.categorized_collection.update_one(
            {"_id": sms["_id"]},
            {"$setOnInsert": categorized_data},
            upsert=True
        )
        
        print(f"✅ Processed SMS: {transaction_type} {amount} to {recipient} → {category}")
    
    def update_combined_transactions(self):
        self.combined_collection.delete_many({})
        category_totals = {}

        # Process SMS transactions
        for doc in self.categorized_collection.find({}):
            original_cat = doc.get("Category", "Miscellaneous")
            if original_cat in self.category_mapping:
                new_cat = self.category_mapping[original_cat]
                amt = float(doc.get("Amount", 0))
                category_totals[new_cat] = category_totals.get(new_cat, 0) + amt

        # Process manual entries
        for doc in self.manual_collection.find({}):
            original_cat = doc.get("expenseName", "Miscellaneous")
            if original_cat in self.category_mapping:
                new_cat = self.category_mapping[original_cat]
                amt = float(doc.get("expenseAmount", 0))
                category_totals[new_cat] = category_totals.get(new_cat, 0) + amt

        # Update combined collection
        for category, total in category_totals.items():
            self.combined_collection.update_one(
                {"_id": category},
                {"$set": {"category": category, "value": round(total, 2)}},
                upsert=True
            )
        print("✅ Updated combined transactions")
    
    def process_new_data(self):
        """Process any new data since last check"""
        # Process manual entries first
        manual_ids = {doc["_id"] for doc in self.manual_collection.find({}, {"_id": 1})}
        new_manual_ids = manual_ids - self.processed_ids
        if new_manual_ids:
            print(f"🆕 Processing {len(new_manual_ids)} new manual entries")
            self.processed_ids.update(new_manual_ids)
            self.update_combined_transactions()
        
        # Process SMS messages
        new_sms = list(self.raw_collection.find({"_id": {"$nin": list(self.processed_ids)}}))
        if new_sms:
            print(f"📨 Processing {len(new_sms)} new SMS messages")
            for sms in new_sms:
                if self.process_single_sms(sms):
                    self.processed_ids.add(sms["_id"])
            self.update_combined_transactions()
        
        return len(new_manual_ids) + len(new_sms)