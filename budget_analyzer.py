# budget_analyzer.py
import pymongo
import pandas as pd
from datetime import datetime

class BudgetAnalyzer:
    def __init__(self, db_client):
        self.db = db_client["expenseDB"]
        self.goals_collection = self.db["goals"]
        self.categorized_collection = self.db["categorized_transactions"]
        self.recommendation_collection = self.db["aicategories"]
        self.insights_collection = self.db["airecommendations"]
        
        self.excluded_categories = {"Healthcare", "Education", "Essentials"}
        self.reducible_categories = {
            "Entertainment & Lifestyle": 0.8,
            "Food & Dining": 0.9,
            "Shopping": 0.8,
            "Miscellaneous": 0.7,
            "Subscriptions & Services": 0.9,
            "Travel & Transportation": 0.9,
        }
    
    def run_monthly_analysis(self):
        print("📊 Running monthly budget analysis...")
        try:
            goals = self.get_user_goals()
            monthly_spending = self.analyze_historical_spending()
            budget_constraints = self.set_budget_constraints(monthly_spending)
            total_savings = self.calculate_savings_potential(monthly_spending, budget_constraints)
            allocated_savings = self.allocate_savings_to_goals(total_savings, goals)
            current_spending = self.get_current_spending()
            
            self.save_recommendations(current_spending, budget_constraints)
            self.generate_insights(current_spending, budget_constraints, total_savings, allocated_savings, goals)
            
            print("✅ Monthly analysis completed successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error in monthly analysis: {e}")
            return False
    
    def get_user_goals(self):
        goal_docs = self.goals_collection.find()
        goals = {}
        for goal in goal_docs:
            try:
                if goal["goalName"] not in self.excluded_categories:
                    goals[goal["_id"]] = {
                        "goalName": goal["goalName"],
                        "targetAmount": float(goal["targetAmount"]),
                        "timeframe": int(goal["timeframe"])
                    }
            except (ValueError, KeyError, TypeError) as e:
                print(f"Skipping invalid goal entry: {goal}, Error: {e}")
        return goals
    
    def analyze_historical_spending(self):
        df = pd.read_csv("Cleaned_Classified.csv")
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df[df['Category'] != "Savings & Transfers"]
        df = df[~df['Category'].isin(self.excluded_categories)]
        monthly_spending = df.groupby('Category')['Amount'].sum() / (df['Date'].nunique() / 30)
        return monthly_spending.to_dict()
    
    def set_budget_constraints(self, monthly_spending):
        constraints = {}
        for category, amount in monthly_spending.items():
            if category in self.excluded_categories:
                continue
            constraints[category] = amount * self.reducible_categories.get(category, 1.0)
        return constraints
    
    def calculate_savings_potential(self, monthly_spending, budget_constraints):
        savings = {
            cat: monthly_spending[cat] - budget_constraints.get(cat, monthly_spending[cat])
            for cat in monthly_spending
            if cat not in self.excluded_categories
        }
        return sum(savings.values())
    
    def allocate_savings_to_goals(self, total_savings, goals):
        valid_goals = {
            _id: g for _id, g in goals.items()
            if g.get("timeframe", 0) > 0
        }
        total_monthly_target = sum(g["targetAmount"] / g["timeframe"] for g in valid_goals.values())
        
        if total_monthly_target == 0:
            return {}
            
        allocated = {}
        for _id, g in valid_goals.items():
            monthly_target = g["targetAmount"] / g["timeframe"]
            allocated[g["goalName"]] = (monthly_target / total_monthly_target) * total_savings
        return allocated
    
    def get_current_spending(self):
        data = self.categorized_collection.find()
        current = {}
        for item in data:
            if item["Transaction Type"] == "debited":
                cat = item["Category"]
                if cat in self.excluded_categories:
                    continue
                try:
                    amt = float(item["Amount"])
                    current[cat] = current.get(cat, 0) + amt
                except:
                    continue
        return current
    
    def save_recommendations(self, current, recommended):
        self.recommendation_collection.delete_many({})
        for category in recommended:
            self.recommendation_collection.insert_one({
                "_id": category,
                "category": category,
                "current": round(current.get(category, 0), 2),
                "recommended": round(recommended[category], 2)
            })
    
    def generate_insights(self, current, recommended, total_savings, allocated, goals):
        insights = []

        # Savings Summary
        insights.append({
            "_id": "summary", 
            "title": "Total Savings Potential", 
            "description": f"₹{total_savings:.2f}/month"
        })

        # Allocated savings
        allocated_lines = [f"{goal}: ₹{amt:.2f}/month" for goal, amt in allocated.items()]
        insights.append({
            "_id": "allocated", 
            "title": "Allocated Savings", 
            "description": "\n".join(allocated_lines) if allocated_lines else "No goals set"
        })

        # Savings Simulation
        sim_lines = []
        for goal_name, allocated_amt in allocated.items():
            goal = next((g for g in goals.values() if g["goalName"] == goal_name), None)
            if goal:
                monthly_target = goal["targetAmount"] / goal["timeframe"]
                if allocated_amt < monthly_target:
                    sim_lines.append(
                        f"You need to save more to meet your {goal_name} goal. "
                        f"(₹{allocated_amt:.2f}/month vs ₹{monthly_target:.2f}/month)"
                    )
        if sim_lines:
            insights.append({
                "_id": "simulation", 
                "title": "Savings Simulation", 
                "description": "\n".join(sim_lines)
            })

        # Actionable Recommendations
        recs = []
        for goal_name, allocated_amt in allocated.items():
            goal = next((g for g in goals.values() if g["goalName"] == goal_name), None)
            if goal:
                monthly_target = goal["targetAmount"] / goal["timeframe"]
                diff = monthly_target - allocated_amt
                if diff > 0:
                    recs.append(
                        f"- To meet your {goal_name} goal, you need to save an additional ₹{diff:.2f}/month."
                    )
        if recs:
            recs.append("- Consider reducing spending in non-essential categories or increasing your income.")
            insights.append({
                "_id": "recommendations", 
                "title": "Recommendations", 
                "description": "\n".join(recs)
            })
        
        self.insights_collection.delete_many({})
        self.insights_collection.insert_many(insights)