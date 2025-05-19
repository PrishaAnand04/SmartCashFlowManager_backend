import pymongo
import re
import pandas as pd
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# MongoDB Setup
connection_string = "mongodb://localhost:27017/"
client = pymongo.MongoClient(connection_string)
db = client["expenseDB"]
goals_collection = db["goals"]
categorized_collection = db["categorized_transactions"]
recommendation_collection = db["aicategories"]
insights_collection = db["airecommendations"]

# Load dataset
df = pd.read_csv("Cleaned_Classified.csv")

# ---------- Constants ---------- #
excluded_categories = {"Healthcare", "Education", "Essentials"}

# ---------- ML Model Setup ---------- #
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
label_encoder = LabelEncoder()
model = None

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.lower()

def initialize_model():
    global model, vectorizer, label_encoder
    try:
        with open('transaction_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('transaction_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('transaction_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except:
        print("Training fresh model...")
        df['cleaned_body'] = df['Body'].apply(clean_text)
        y = label_encoder.fit_transform(df['Category'])
        X = vectorizer.fit_transform(df['cleaned_body'])
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        with open('transaction_classifier.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('transaction_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('transaction_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

# ---------- Budgeting Logic ---------- #
def get_user_goals():
    goal_docs = goals_collection.find()
    goals = {}
    for goal in goal_docs:
        try:
            if goal["goalName"] not in excluded_categories:
                goals[goal["_id"]] = {
                    "goalName": goal["goalName"],
                    "targetAmount": float(goal["targetAmount"]),
                    "timeframe": int(goal["timeframe"])
                }
        except (ValueError, KeyError, TypeError) as e:
            print(f"Skipping invalid goal entry: {goal}, Error: {e}")
    return goals

def analyze_spending(df):
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df[df['Category'] != "Savings & Transfers"]
    df = df[~df['Category'].isin(excluded_categories)]
    monthly_spending = df.groupby('Category')['Amount'].sum() / (df['Date'].nunique() / 30)
    return monthly_spending.to_dict()

def set_budget_constraints(monthly_spending):
    reducible_categories = {
        "Entertainment & Lifestyle": 0.8,
        "Food & Dining": 0.9,
        "Shopping": 0.8,
        "Miscellaneous": 0.7,
        "Subscriptions & Services": 0.9,
        "Travel & Transportation": 0.9,
    }
    budget_constraints = {}
    for category, amount in monthly_spending.items():
        if category in excluded_categories:
            continue
        elif category in reducible_categories:
            budget_constraints[category] = amount * reducible_categories[category]
        else:
            budget_constraints[category] = amount
    return budget_constraints

def calculate_savings_potential(monthly_spending, budget_constraints):
    savings_potential = {
        cat: monthly_spending[cat] - budget_constraints.get(cat, monthly_spending[cat])
        for cat in monthly_spending
        if cat not in excluded_categories
    }
    return sum(savings_potential.values())

def allocate_savings_to_goals(total_savings_potential, goals):
    valid_goals = {
        _id: g for _id, g in goals.items()
        if g.get("timeframe", 0) > 0
    }
    total_monthly_target = sum(g["targetAmount"] / g["timeframe"] for g in valid_goals.values())
    allocated_savings = {}
    for _id, g in valid_goals.items():
        monthly_target = g["targetAmount"] / g["timeframe"]
        allocated_savings[g["goalName"]] = (monthly_target / total_monthly_target) * total_savings_potential
    return allocated_savings

def get_current_spending():
    data = categorized_collection.find()
    current_spending = {}
    for item in data:
        if item["Transaction Type"] == "debited":
            cat = item["Category"]
            if cat in excluded_categories:
                continue
            try:
                amt = float(item["Amount"])
                current_spending[cat] = current_spending.get(cat, 0) + amt
            except:
                continue
    return current_spending

# ---------- AI Insights Generator ---------- #
def generate_ai_insights(current, recommended, total_savings, allocated, goals):
    insights = []

    # Line 1: Savings Summary
    insights.append({"_id": "summary", "title": "Total Savings Potential", "description": f"₹{total_savings:.2f}/month"})

    # Line 2: Allocated savings
    allocated_lines = [f"{goal}: ₹{amt:.2f}/month" for goal, amt in allocated.items()]
    insights.append({"_id": "allocated", "title": "Allocated Savings", "description": "\n".join(allocated_lines)})

    # Line 3: Savings Simulation
    sim_lines = []
    for goal_name, allocated_amt in allocated.items():
        goal = next((g for g in goals.values() if g["goalName"] == goal_name), None)
        if goal:
            monthly_target = goal["targetAmount"] / goal["timeframe"]
            if allocated_amt < monthly_target:
                sim_lines.append(f"You need to save more to meet your {goal_name} goal. (₹{allocated_amt:.2f}/month vs ₹{monthly_target:.2f}/month)")
    if sim_lines:
        insights.append({"_id": "simulation", "title": "Savings Simulation", "description": "\n".join(sim_lines)})

    # Line 4: Actionable Recommendations
    recs = []
    for goal_name, allocated_amt in allocated.items():
        goal = next((g for g in goals.values() if g["goalName"] == goal_name), None)
        if goal:
            monthly_target = goal["targetAmount"] / goal["timeframe"]
            diff = monthly_target - allocated_amt
            if diff > 0:
                recs.append(f"- To meet your {goal_name} goal, you need to save an additional ₹{diff:.2f}/month.")
    if recs:
        recs.append("- Consider reducing spending in non-essential categories or increasing your income.")
        insights.append({"_id": "recommendations", "title": "Recommendations", "description": "\n".join(recs)})

    return insights

# ---------- Main Execution ---------- #
def run_monthly_analysis():
    initialize_model()

    # Step 1: Get goals
    goals = get_user_goals()

    # Step 2: Analyze historical spending
    monthly_spending = analyze_spending(df)

    # Step 3: Set constraints
    budget_constraints = set_budget_constraints(monthly_spending)

    # Step 4: Calculate savings
    total_savings_potential = calculate_savings_potential(monthly_spending, budget_constraints)

    # Step 5: Allocate to goals
    allocated_savings = allocate_savings_to_goals(total_savings_potential, goals)

    # Step 6: Real-time spending
    current_spending = get_current_spending()

    # Step 7: Save to MongoDB
    recommendation_collection.delete_many({})
    for category in budget_constraints:
        recommendation_collection.insert_one({
            "_id": category,
            "category": category,
            "current": round(current_spending.get(category, 0), 2),
            "recommended": round(budget_constraints[category], 2)
        })

    # Step 8: AI Insights
    insights = generate_ai_insights(current_spending, budget_constraints, total_savings_potential, allocated_savings, goals)
    insights_collection.delete_many({})
    insights_collection.insert_many(insights)

    print("✅ Monthly financial recommendation and insights updated in MongoDB.")

# Run
if __name__ == "__main__":
    run_monthly_analysis()
