const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect('mongodb://localhost:27017/expenseDB', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

/***** Original Schemas *****/
// Expense Schema
const expenseSchema = new mongoose.Schema({
  expenseName: String,
  expenseMonth: String,
  expenseAmount: String,
  createdAt: { type: Date, default: Date.now }
});

// Goal Schema
const goalSchema = new mongoose.Schema({
  goalName: String,
  targetAmount: String,
  timeframe: String,
  createdAt: { type: Date, default: Date.now }
});

/*****Monthly Chart Schema *****/
const monthlyChartSchema = new mongoose.Schema({
  month: String,
  monthNumber: Number,
  value: Number,
  createdAt: { type: Date, default: Date.now }
});

/*****AI Category schema *****/

const AICategorySchema = new mongoose.Schema({
  category: String,
  current: Number,
  recommended: Number,
  createdAt: { type: Date, default: Date.now }
});

const AIRecommendationSchema = new mongoose.Schema({
  title: String,
  description: String,
  createdAt: { type: Date, default: Date.now }
});

//summary Schema
const summarySchema = new mongoose.Schema({
  category: String,
  amount: Number,
  createdAt: { type: Date, default: Date.now }
});

//category chart schema
const categoryChartSchema = new mongoose.Schema({
  category: String,
  value: Number,
  createdAt: { type: Date, default: Date.now }
});

//sms 
const smsMessageSchema = new mongoose.Schema({
  body: String,
  sender:String,
  createdAt: { type: Date, default: Date.now }
});

const messageSchema = new mongoose.Schema({
  address: String,
  date: String,
  time: String,
  body: String,
});



// Create Models
const Expense = mongoose.model('Expense', expenseSchema);
const Goal = mongoose.model('Goal', goalSchema);
const MonthlyChart = mongoose.model('MonthlyChart', monthlyChartSchema);
const AICategory = mongoose.model('AICategory', AICategorySchema);
const AIRecommendation = mongoose.model('AIRecommendation', AIRecommendationSchema);
const Summary = mongoose.model('Summary', summarySchema); // ✅ NEW
const CategoryChart = mongoose.model('CategoryChart', categoryChartSchema); // ✅ NEW MODEL
const SmsMessage = mongoose.model('SmsMessage', smsMessageSchema);
const Message = mongoose.model('Message', messageSchema);


/***** Original Endpoints *****/
// Expense Endpoints
app.post('/api/addexpense', async (req, res) => {
  try {
    const newExpense = new Expense(req.body);
    await newExpense.save();
    res.status(200).json({ message: "Expense saved to MongoDB!" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/getexpense", async (req, res) => {
  try {
    const expenses = await Expense.find().sort({ createdAt: -1 });
    res.status(200).json({ expenses });
  } catch (error) {
    console.error("Error fetching expenses:", error);
    res.status(500).json({ error: "Failed to fetch expenses" });
  }
});

// Goal Endpoints
app.post('/api/addgoal', async (req, res) => {
  try {
    const newGoal = new Goal(req.body);
    await newGoal.save();
    res.status(200).json({ message: "Goal saved to MongoDB!" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/getgoals", async (req, res) => {
  try {
    const goals = await Goal.find().sort({ createdAt: -1 });
    res.status(200).json({ goals });
  } catch (error) {
    console.error("Error fetching goals:", error);
    res.status(500).json({ error: "Failed to fetch goals" });
  }
});

/***** New Monthly Chart Endpoint *****/
app.get('/api/monthly-expenses', async (req, res) => {
  try {
    const monthlyData = await MonthlyChart.find().sort({ monthNumber: 1 });
    res.json(monthlyData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});


app.get('/api/ai-categories', async (req, res) => {
  try {
    const categories = await AICategory.find().sort({ createdAt: -1 });
    res.json(categories);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Get AI Text Recommendations
app.get('/api/ai-text-recommendations', async (req, res) => {
  try {
    const recommendations = await AIRecommendation.find().sort({ createdAt: -1 });
    res.json(recommendations);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

//summary endpoint
app.get('/api/monthly-summary', async (req, res) => {
  try {
    const summaryData = await Summary.find().sort({ createdAt: -1 });
    res.json(summaryData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

//category chart endpoint
/***** NEW Category Chart Endpoint *****/
app.get('/api/category-data', async (req, res) => {
  try {
    const categoryData = await CategoryChart.find().sort({ createdAt: -1 });
    res.json(categoryData);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

app.post('/api/save_sms', async (req, res) => {
  const { body, sender } = req.body; // Changed from 'message' to 'body'
  if (!body || !sender) {
    return res.status(400).json({ error: 'Missing body or sender' });
  }
  try {
    const newSms = new SmsMessage({ 
      body: body, 
      sender: sender 
    });
    await newSms.save();
    res.status(200).json({ status: 'success' });
  } catch (error) {
    console.error("Failed to save SMS:", error);
    res.status(500).json({ error: 'Failed to save SMS to DB' });
  }
});

//message
app.post('/api/messages', async (req, res) => {
  try {
    const message = new Message(req.body);
    await message.save();
    res.status(201).json(message);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/messages', async (req, res) => {
  try {
    const messages = await Message.find().sort({ date: -1 });
    res.json(messages);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});