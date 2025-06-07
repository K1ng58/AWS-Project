
# 💰 Personal Finance & Investment Coach – README

## Overview

This is a **Streamlit-based personal finance and investment assistant** that integrates AWS Bedrock’s Claude model and financial data APIs to deliver a comprehensive, intelligent, and interactive dashboard for individuals to manage income, spending, investments, debt, and financial goals. It also supports PDF statement parsing and financial Q&A.

---

## 🛠️ Key Features

### 1. **Streamlit Frontend**
- Intuitive dashboard-style interface
- Sidebar navigation across all modules
- Supports mock data initialization and user data resets

### 2. **Modular Pages**
- **Dashboard** – Financial summary, goals snapshot, and key metrics
- **Goals** – Add, view, and track financial goals with progress and projections
- **Budget** – Analyze categorized expenses, trends, and upload credit card PDFs
- **Investments** – View portfolio, simulate returns, get ETF recommendations
- **Debt Management** – Track liabilities and get strategy suggestions (snowball vs avalanche)
- **Financial Q&A** – Ask questions and receive LLM-powered answers based on user profile
- **Upload Statement** – Extract and classify credit card transactions from PDFs
- **Market News** – Pull real-time financial news via yfinance, RSS, and scraping

### 3. **AI Integration**
- AWS Bedrock with Claude v3.5 (Sonnet) for natural language processing
- ConversationalRetrievalChain + FAISS-powered vector search for financial document Q&A

### 4. **Data Handling**
- Simulated mock data for income, expenses, investments, and goals
- Cleaned category labels for consistent analysis
- Uses yfinance to fetch stock prices and portfolio news

---

## 🔧 How to Run

### Prerequisites
- Python 3.9+
- AWS credentials with Bedrock access
- `.env` file with:
  ```
  AWS_ACCESS_KEY_ID=your_key
  AWS_SECRET_ACCESS_KEY=your_secret
  AWS_REGION=us-west-2
  ```
- Install dependencies:
  ```bash
  pip install streamlit pandas yfinance matplotlib plotly boto3 langchain faiss-cpu python-dotenv feedparser PyPDF2
  ```

### Launch the App
```bash
streamlit run AWS\ Final\ GITHUB.py
```

---

## 📂 Project Structure
- `FinancialAgent` class: manages all user data and computations
- `main()`: Streamlit app logic with conditional routing
- Claude/Bedrock logic for question answering and investment advice
- PDF transaction parser tailored for credit card statements

---

## 💡 Tips
- First-time load uses mock data – clear and reupload for real use
- PDF parsing is tuned to Discover-like statements but is extendable
- Claude-based Q&A enriches answers with user context and embeddings

---

## 📌 Limitations
- This is a prototype – no database or persistent storage
- Claude may not answer if AWS keys are invalid or Bedrock is unreachable
- Not intended for production without security and authentication layers

---

## 📞 Support
Developed by: **David G (Cal Poly)**  
Questions? Reach out via your project channel.
