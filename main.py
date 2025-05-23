# main.py
import pymongo
import time
import threading
from datetime import datetime
from transaction_processor import TransactionProcessor
from budget_analyzer import BudgetAnalyzer

class FinancialManager:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.processor = TransactionProcessor(self.client)
        self.analyzer = BudgetAnalyzer(self.client)
        self.last_sms_count = 0
        self.last_manual_count = 0
        self.lock = threading.Lock()
        
    def run(self):
        print("🚀 Starting Financial Management System")
        
        # Initial counts
        self.last_sms_count = self.processor.raw_collection.count_documents({})
        self.last_manual_count = self.processor.manual_collection.count_documents({})
        
        # Start real-time processing in background
        processing_thread = threading.Thread(target=self.monitor_changes, daemon=True)
        processing_thread.start()
        
        # Also run monthly analysis periodically
        self.run_monthly_analysis_loop()
    
    def monitor_changes(self):
        """Monitor for new transactions and trigger processing"""
        while True:
            try:
                current_sms_count = self.processor.raw_collection.count_documents({})
                current_manual_count = self.processor.manual_collection.count_documents({})
                
                if (current_sms_count != self.last_sms_count or 
                    current_manual_count != self.last_manual_count):
                    
                    with self.lock:
                        print("🔄 New data detected - processing transactions...")
                        self.processor.process_new_data()
                        self.last_sms_count = current_sms_count
                        self.last_manual_count = current_manual_count
                        
                        print("📊 Running budget analysis...")
                        self.analyzer.run_monthly_analysis()
                
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(10)
    
    def run_monthly_analysis_loop(self):
        """Run monthly analysis at the start of each month"""
        last_analysis_month = None
        while True:
            current_month = datetime.now().month
            if current_month != last_analysis_month:
                with self.lock:
                    print(f"📅 New month detected ({current_month}) - running full analysis")
                    self.analyzer.run_monthly_analysis()
                    last_analysis_month = current_month
            
            time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    manager = FinancialManager()
    manager.run()