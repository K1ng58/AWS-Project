# ─────────────────────────────────────────────
# 📦 Standard Library Imports
# ─────────────────────────────────────────────
import os
import io
import re
import glob
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# ─────────────────────────────────────────────
# 🌐 Third-party Libraries
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2
import feedparser
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 🤖 LangChain Core & AWS Bedrock
# ─────────────────────────────────────────────
import boto3
from langchain import hub
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage, HumanMessage

# ─────────────────────────────────────────────
# 🔍 LangChain Tools
# ─────────────────────────────────────────────
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsInput
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment variables from .env file
load_dotenv()

# Placeholder for `llm`. In your actual application, you'd define or initialize this.
llm = None 

# AWS Bedrock client setup
client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("xxx"), # Replace with your AWSKEYS
    aws_secret_access_key=os.getenv("xxx"), 
    region_name=os.getenv("AWS_REGION", "us-west-2")
)

# Initial Bedrock invocation (for testing connection)
try:
    # Changed to BedrockChat for Claude 3 Sonnet
    response = client.invoke_model(
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "Hello Claude, are you there?"}]
        }),
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    # print(response["body"].read().decode()) # Uncomment to see the test response
except Exception as e:
    st.error(f"Error initializing Bedrock client or invoking model: {e}")


# Claude LLM wrapper (assuming aws_client is defined elsewhere or should be 'client')
def query_claude(prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    accept = "application/json"
    content_type = "application/json"

    try:
        # Assuming 'client' is the correct boto3 client, not 'aws_client'
        response = client.invoke_model( 
            body=json.dumps(body),
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"].strip()
    except Exception as e:
        return f"[ERROR] Claude query failed: {e}"

# Utility function for compound interest
def compound_interest(principal, rate, years, contributions=0, contribution_freq="monthly"):
    """Calculate compound interest with regular contributions"""
    total = principal
    monthly_rate = rate / 12
    
    for month in range(int(years * 12)):
        if contribution_freq == "monthly":
            total += contributions
        elif contribution_freq == "annual" and month % 12 == 0 and month > 0:
            total += contributions
            
        total *= (1 + monthly_rate / 12)
        
    return total

def calculate_budget_allocation(income, expenses_dict):
    """Calculate budget allocation percentages"""
    total_expenses = sum(expenses_dict.values())
    savings = income - total_expenses
    
    allocations = {category: (amount / income) * 100 for category, amount in expenses_dict.items()}
    allocations["Savings"] = (savings / income) * 100
    
    return allocations

def risk_profile_to_allocation(risk_profile: str) -> dict:
    risk_profile = risk_profile.lower()
    if risk_profile == "conservative":
        return {"Stocks": 0.3, "Bonds": 0.6, "Cash": 0.1, "Alternative": 0.0}
    elif risk_profile == "moderate":
        return {"Stocks": 0.5, "Bonds": 0.4, "Cash": 0.05, "Alternative": 0.05}
    elif risk_profile == "aggressive":
        return {"Stocks": 0.8, "Bonds": 0.1, "Cash": 0.05, "Alternative": 0.05}
    else: # Default or Balanced
        return {"Stocks": 0.6, "Bonds": 0.3, "Cash": 0.05, "Alternative": 0.05}

def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to clean and standardize categories
def clean_category(category_name: str) -> str:
    category_name = str(category_name).lower().strip() # Ensure it's a string
    if "restaurant" in category_name or "dining" in category_name or "eats" in category_name or "cafe" in category_name:
        return "Restaurants"
    if "grocery" in category_name or "supermarket" in category_name or "food" in category_name:
        return "Groceries"
    if "gasoline" in category_name or "fuel" in category_name:
        return "Gasoline"
    if "merchandise" in category_name or "store" in category_name or "shopping" in category_name or "retail" in category_name:
        return "Shopping/Merchandise"
    if "entertainment" in category_name or "movie" in category_name or "game" in category_name or "leisure" in category_name:
        return "Entertainment"
    if "utility" in category_name or "electric" in category_name or "water" in category_name or "internet" in category_name:
        return "Utilities"
    if "transportation" in category_name or "transit" in category_name or "travel" in category_name:
        return "Transportation"
    if "housing" in category_name or "rent" in category_name or "mortgage" in category_name:
        return "Housing"
    # Add more mappings as needed based on your statement data
    return category_name.title() # Capitalize first letter of unmatched categories

def create_mock_transaction_data():
    """Create mock transaction data for demonstration"""
    # Updated categories to reflect expected cleaned categories
    categories = ["Groceries", "Restaurants", "Housing", "Transportation", "Entertainment", "Utilities", "Shopping/Merchandise"]
    
    # Generate mock transactions for the past 3 months
    today = datetime.now()
    start_date = today - timedelta(days=90)
    
    transactions = []
    
    # Generate random transactions
    for i in range(100):
        date = start_date + timedelta(days=random.randint(0, 90))
        category = random.choice(categories)
        
        # Set realistic amounts based on category
        if category == "Housing":
            amount = random.uniform(800, 2000)
        elif category in ["Groceries", "Utilities"]:
            amount = random.uniform(50, 300)
        elif category == "Transportation":
            amount = random.uniform(30, 200)
        else:
            amount = random.uniform(10, 150)
            
        transactions.append({
            "date": date.strftime("%Y-%m-%d"),
            "category": category, # Use the already cleaned category here
            "amount": round(amount, 2),
            "description": f"Sample {category} expense"
        })
    
    return pd.DataFrame(transactions)

def create_mock_income_data():
    """Create mock income data for demonstration"""
    today = datetime.now()
    start_date = today - timedelta(days=90)
    
    income = []
    # Bi-weekly pay
    current_date = start_date
    while current_date <= today:
        income.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "source": "Salary",
            "amount": 2500.00,
            "description": "Bi-weekly paycheck"
        })
        current_date += timedelta(days=14)
    
    # Random additional income
    for i in range(3):
        date = start_date + timedelta(days=random.randint(0, 90))
        amount = random.uniform(50, 500)
        income.append({
            "date": (start_date + timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"), # Corrected line
            "source": "Side Gig",
            "amount": round(amount, 2),
            "description": "Freelance work"
        })
    
    return pd.DataFrame(income)

def setup_financial_knowledge_base():
    """Set up vector database for financial knowledge"""
    # In a real application, you would load actual financial documents
    # For this demo, we'll create a small sample
    
    financial_texts = [
        "Compound interest is the addition of interest to the principal sum of a loan or deposit. It is the result of reinvesting interest, rather than paying it out, so that interest in the next period is earned on the principal sum plus previously accumulated interest.",
        "Dollar-cost averaging (DCA) is an investment strategy in which an investor divides up the total amount to be invested across periodic purchases of a target asset in an effort to reduce the impact of volatility on the overall purchase.",
        "A 401(k) is a tax-advantaged, defined-contribution retirement account offered by many employers to their employees. Workers can make contributions to their 401(k) accounts through automatic payroll withholding, and their employers can match some or all of those contributions.",
        "An emergency fund is a financial safety net that everyone should have for life's unforeseen expenses. The purpose of an emergency fund is to improve financial security by creating a safety net that can be used to meet unanticipated expenses.",
        "Asset allocation is an investment strategy that aims to balance risk and reward by apportioning a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon.",
        "Market timing is the strategy of making buying or selling decisions of financial assets by attempting to predict future market price movements."
    ]

    try:
        global llm
        if llm is None:
            # Use BedrockChat for Claude 3 models
            llm = BedrockChat(
                client=client, 
                model_id="anthropic.claude-3-sonnet-20240229-v1:0" 
            ) 
        
        embeddings = BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v1")


        if not os.path.exists("temp_docs"):
            st.warning("Knowledge base directory 'temp_docs' not found. Creating dummy documents for knowledge base.")
            os.makedirs("temp_docs", exist_ok=True)
            for i, text in enumerate(financial_texts):
                with open(f"temp_docs/doc_{i}.txt", "w") as f:
                    f.write(text)
            
        loader = DirectoryLoader("temp_docs", glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)

        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        return retrieval_chain
    except Exception as e:
        st.error(f"Error setting up financial knowledge base: {e}")
        return None

# New function to parse credit card statements
def parse_credit_card_statement(uploaded_file):
    transactions = []
    if uploaded_file is not None:
        try:
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Regex to find transaction lines. This pattern is specific to your provided PDF.
                # It looks for a date (MM/DD), then description (can be multiple words/lines),
                # and finally an amount with optional leading signs.
                # Adjust this regex based on the actual format of your statements.
                # Example: "04/23         GAMESTOP 8008838895 TX      Merchandise         $34.97"
                # This regex needs to be robust for various transaction descriptions.
                
                # Improved regex to capture date, description, category, and amount more reliably
                # This regex assumes the merchant category is always a single word
                # and the amount is always at the end of the line.
                transaction_pattern = re.compile(
                    r'(\d{2}/\d{2})\s+([A-Z0-9\s\.\*#\-\(\)]+?)\s+([A-Za-z\s\/]+?)\s+\$([\d\.,]+)'
                )
                
                # Look for the "PURCHASES" section to start extraction
                if "PURCHASES" in text:
                    lines = text.split('\n')
                    in_purchases_section = False
                    for line in lines:
                        if "PURCHASES" in line:
                            in_purchases_section = True
                            continue # Skip the header line itself
                        
                        if in_purchases_section:
                            # Stop if we hit another major section like "Cashback Bonus® Rewards"
                            if "Cashback Bonus® Rewards" in line or "PREVIOUS BALANCE" in line:
                                in_purchases_section = False
                                break
                            
                            match = transaction_pattern.search(line.strip())
                            if match:
                                trans_date_str, description, category, amount_str = match.groups()
                                
                                try:
                                    # Construct a full date for the current year
                                    current_year = datetime.now().year
                                    trans_date = datetime.strptime(f"{trans_date_str}/{current_year}", "%m/%d/%Y").strftime("%Y-%m-%d")
                                    amount = float(amount_str.replace(',', ''))
                                    
                                    # Apply category cleaning here
                                    cleaned_category = clean_category(category)

                                    transactions.append({
                                        "date": trans_date,
                                        "category": cleaned_category, # Use the cleaned category
                                        "amount": amount,
                                        "description": description.strip()
                                    })
                                except ValueError as ve:
                                    st.warning(f"Could not parse line: '{line.strip()}' - {ve}")
                                    continue # Skip malformed lines
            
            if transactions:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(transactions)
                # Filter out any zero amount transactions or similar anomalies if they occur
                df = df[df['amount'] > 0]
                st.success(f"Successfully extracted {len(df)} transactions from the PDF.")
                return df
            else:
                st.warning("No transactions found in the PDF using the defined pattern.")
                return pd.DataFrame()

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


class FinancialAgent:
    def __init__(self):
        self.user_data = {
            "personal_info": {},
            "income": pd.DataFrame(), # Initialize as empty DataFrame
            "expenses": pd.DataFrame(), # Initialize as empty DataFrame
            "goals": [],
            "risk_profile": "moderate",
            "investment_portfolio": {},
            "chat_history": []
        }
        self.knowledge_base = None
        self._mock_data_initialized_on_start = False # New flag to control mock data loading
        self.load_user_data() # Ensure mock data is loaded on initialization
        
    def load_user_data(self):
        """Load user data from file or create mock data if not available (only on initial load or full clear)"""

        # Only load mock data if it hasn't been initialized on app start OR
        # if clear_all_data was just called and reset _mock_data_initialized_on_start to False.
        if not self._mock_data_initialized_on_start:
            # Personal information
            self.user_data["personal_info"] = {
                "age": 28,
                "income_yearly": 75000,
                "tax_rate": 0.25,
                "debt": { # Default debt for simulation
                    "student_loans": 15000,
                    "credit_card": 2000,
                    "car_loan": 8000
                },
                "savings_rate": 0.15
            }
            # Mock DataFrames
            self.user_data["expenses"] = create_mock_transaction_data()
            self.user_data["income"] = create_mock_income_data()
            
            # Example goals
            self.user_data["goals"] = [
                {
                    "name": "Emergency Fund",
                    "target_amount": 15000,
                    "current_amount": 5000,
                    "monthly_contribution": 500,
                    "target_date": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
                    "priority": "High"
                },
                {
                    "name": "House Down Payment",
                    "target_amount": 60000,
                    "current_amount": 12000,
                    "monthly_contribution": 800,
                    "target_date": (datetime.now() + timedelta(days=1095)).strftime("%Y-%m-%d"),
                    "priority": "Medium"
                },
                {
                    "name": "Retirement",
                    "target_amount": 1000000,
                    "current_amount": 25000,
                    "monthly_contribution": 1000,
                    "target_date": (datetime.now() + timedelta(days=12775)).strftime("%Y-%m-%d"),
                    "priority": "Medium"
                }
            ]
            
            # Example portfolio
            self.user_data["investment_portfolio"] = {
                "assets": [
                    {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "shares": 20, "purchase_price": 205.75, "asset_type": "Stock ETF"},
                    {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF", "shares": 15, "purchase_price": 56.82, "asset_type": "Stock ETF"},
                    {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "shares": 10, "purchase_price": 72.54, "asset_type": "Bond ETF"},
                    {"ticker": "BNDX", "name": "Vanguard Total International Bond ETF", "shares": 5, "purchase_price": 48.65, "asset_type": "Bond ETF"}
                ],
                "cash": 5000.00 # Default cash amount
            }
            
            # Risk profile questionnaire results
            self.user_data["risk_profile"] = "moderate"
            
            self._mock_data_initialized_on_start = True # Set flag after initial load
        
    def save_user_data(self):
        """Save user data to storage"""
        # In a real app, save to S3 or local storage
        pass

    def clear_all_data(self):
        """Clears all user data (expenses, income, goals, portfolio, personal info)"""
        self.user_data = {
            "personal_info": {}, # Will be repopulated by load_user_data
            "income": pd.DataFrame(),
            "expenses": pd.DataFrame(),
            "goals": [],
            "risk_profile": "moderate", 
            "investment_portfolio": {},
            "chat_history": []
        }
        self._mock_data_initialized_on_start = False # Reset flag to force mock data reload
        self.load_user_data() # Reload mock data after clearing to ensure fields are populated
        st.success("All user data cleared!")

    def clear_expenses(self):
        self.user_data["expenses"] = pd.DataFrame()
        st.success("Expense data cleared!")
        # Do NOT call self.load_user_data() here. Let main() and st.rerun() handle it.

    def clear_income(self):
        self.user_data["income"] = pd.DataFrame()
        st.success("Income data cleared!")
        # Do NOT call self.load_user_data() here.

    def clear_goals(self):
        self.user_data["goals"] = []
        st.success("Financial goals cleared!")
        # Do NOT call self.load_user_data() here.

    def clear_investment_portfolio(self):
        self.user_data["investment_portfolio"] = {"assets": [], "cash": 5000} # Reset cash to a valid default
        st.success("Investment portfolio data cleared!")
        # Do NOT call self.load_user_data() here.

    def clear_personal_info_debt(self):
        if "personal_info" in self.user_data:
            self.user_data["personal_info"]["debt"] = {}
        else:
            self.user_data["personal_info"] = {"debt": {}} # Ensure personal_info exists
        st.success("Debt data cleared!")
        # Do NOT call self.load_user_data() here.
    
    def get_monthly_income(self):
        """Calculate total monthly income"""
        if isinstance(self.user_data["income"], pd.DataFrame) and not self.user_data["income"].empty:
            income_df = self.user_data["income"].copy()
            income_df["date"] = pd.to_datetime(income_df["date"])
            # Get the number of unique months in the income data
            num_months = income_df["date"].dt.to_period('M').nunique()
            if num_months > 0:
                return income_df["amount"].sum() / num_months
            return 0 # Avoid division by zero if no income data
        # Fallback if no income data, but personal_info might have a yearly estimate
        return self.user_data["personal_info"].get("income_yearly", 0) / 12
    
    def get_monthly_expenses_by_category(self):
        """Get monthly expenses grouped by category"""
        if isinstance(self.user_data["expenses"], pd.DataFrame) and not self.user_data["expenses"].empty:
            expenses_df = self.user_data["expenses"].copy()
            expenses_df["date"] = pd.to_datetime(expenses_df["date"])
            
            # Apply category cleaning to all expenses before grouping
            expenses_df["category"] = expenses_df["category"].apply(clean_category)

            # Group by category and calculate sum
            grouped = expenses_df.groupby("category")["amount"].sum().reset_index()
            
            # Get the number of unique months in the expenses data
            num_months = expenses_df["date"].dt.to_period('M').nunique()
            
            if num_months > 0:
                grouped["amount"] = grouped["amount"] / num_months
            else:
                grouped["amount"] = 0 # If no months, amounts are 0
            
            return dict(zip(grouped["category"], grouped["amount"]))
        return {}
    
    def calculate_net_worth(self):
        """Calculate user's net worth"""
        # Assets
        investment_value = self.get_investment_portfolio_value()
        cash = self.user_data["investment_portfolio"].get("cash", 0)
        # Sum current_amount from all goals (assuming goals are assets)
        total_goal_current_amount = sum(goal.get("current_amount", 0) for goal in self.user_data["goals"])

        # Liabilities
        debt = sum(self.user_data["personal_info"].get("debt", {}).values())
        
        return (investment_value + cash + total_goal_current_amount) - debt
    
    def get_investment_portfolio_value(self):
        """Calculate current value of investment portfolio, fetching live prices with yfinance."""
        total_value = self.user_data["investment_portfolio"].get("cash", 0.0)
        for asset in self.user_data["investment_portfolio"].get("assets", []):
            ticker = asset.get("ticker")
            shares = asset.get("shares", 0)
            purchase_price = asset.get("purchase_price", 0)
            if not ticker or shares <= 0: continue
            try:
                stock_info = yf.Ticker(ticker)
                hist = stock_info.history(period="1d")
                current_price = hist["Close"].iloc[-1] if not hist.empty else stock_info.info.get('currentPrice', purchase_price)
                if current_price is None or current_price == 0: current_price = purchase_price # Fallback if live data is bad
                asset_value = current_price * shares
                total_value += asset_value
            except Exception as e:
                # st.warning(f"Error fetching live data for {ticker}: {e}. Using purchase price (${purchase_price}) as fallback.")
                total_value += purchase_price * shares
        return total_value    
    '''
    def get_investment_portfolio_value(self):
        """Calculate current value of investment portfolio"""
        total_value = 0
        
        for asset in self.user_data["investment_portfolio"].get("assets", []):
            ticker = asset["ticker"]
            shares = asset["shares"]
            
            try:
                # Use yfinance to get the latest price
                stock_info = yf.Ticker(ticker)
                # Sometimes history(period="1d") might return an empty dataframe for less liquid assets or API issues
                hist = stock_info.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                else:
                    # Fallback to current price if history is empty (e.g., non-trading day)
                    current_price = stock_info.info.get('currentPrice', asset["purchase_price"])
                    st.warning(f"Could not get 1-day history for {ticker}. Using alternative price source.")

                asset_value = current_price * shares
                total_value += asset_value
            except Exception as e:
                # If API fails, use purchase price as a fallback
                st.warning(f"Error fetching live data for {ticker}: {e}. Using purchase price as fallback.")
                total_value += asset["purchase_price"] * shares
        
        return total_value
    '''
    # Function to add new investments
    def add_investment(self, ticker: str, name: str, shares: float, purchase_price: float, asset_type: str = "ETF"):
        """
        Adds a new investment (stock, bond, ETF, etc.) to the portfolio.
        Does NOT deduct the cost from cash, as requested.
        """
        new_asset = {
            "ticker": ticker.upper(),
            "name": name,
            "shares": shares,
            "purchase_price": purchase_price,
            "asset_type": asset_type
        }
        self.user_data["investment_portfolio"]["assets"].append(new_asset)
        # Removed: self.user_data["investment_portfolio"]["cash"] -= cost # This line is removed
        # Removed: self.user_data["investment_portfolio"]["cash"] = float(self.user_data["investment_portfolio"]["cash"]) # This line is removed
        
        st.success(f"Added {shares} shares of {name} ({ticker}) to your portfolio.")
        return True
    
    def get_detailed_holdings_df(self):
        """
        Returns a DataFrame of current holdings including live price, current value,
        and unrealized gain/loss.
        """
        detailed_assets = []
        for asset in self.user_data["investment_portfolio"].get("assets", []):
            ticker = asset.get("ticker")
            shares = asset.get("shares", 0)
            purchase_price = asset.get("purchase_price", 0)

            current_price = purchase_price # Default to purchase price
            try:
                stock_info = yf.Ticker(ticker)
                hist = stock_info.history(period="1d")
                if not hist.empty and "Close" in hist.columns:
                    current_price = hist["Close"].iloc[-1]
                elif stock_info.info.get('currentPrice') is not None:
                    current_price = stock_info.info.get('currentPrice')
            except Exception:
                pass # Fallback to purchase_price on error

            current_value = current_price * shares
            unrealized_gain_loss = (current_price - purchase_price) * shares

            detailed_assets.append({
                "Ticker": ticker,
                "Name": asset.get("name"),
                "Shares": shares,
                "Purchase Price": f"${purchase_price:,.2f}",
                "Current Price": f"${current_price:,.2f}",
                "Current Value": f"${current_value:,.2f}",
                "Unrealized Gain/Loss": f"${unrealized_gain_loss:,.2f}",
                "Asset Type": asset.get("asset_type")
            })
        
        return pd.DataFrame(detailed_assets)
    def analyze_spending_trends(self):
        """Analyze spending trends over time"""
        if isinstance(self.user_data["expenses"], pd.DataFrame) and not self.user_data["expenses"].empty:
            expenses_df = self.user_data["expenses"].copy()
            expenses_df["date"] = pd.to_datetime(expenses_df["date"])
            expenses_df["month"] = expenses_df["date"].dt.strftime("%Y-%m")
            
            # Apply category cleaning to all expenses before grouping
            expenses_df["category"] = expenses_df["category"].apply(clean_category)

            monthly_spending = expenses_df.groupby(["month", "category"])["amount"].sum().reset_index()
            return monthly_spending
        return pd.DataFrame()
    
    def track_goal_progress(self, goal_name):
        """Calculate progress towards a specific financial goal"""
        goal = next((g for g in self.user_data["goals"] if g["name"] == goal_name), None)
        
        if not goal:
            return None
        
        current_amount = goal["current_amount"]
        target_amount = goal["target_amount"]
        progress = (current_amount / target_amount) * 100
        
        target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
        remaining_days = (target_date - datetime.now()).days
        
        if remaining_days <= 0:
            on_track = current_amount >= target_amount
        else:
            monthly_contribution = goal.get("monthly_contribution", 0) # Use .get for safety
            projected_amount = current_amount + (monthly_contribution * remaining_days / 30)
            on_track = projected_amount >= target_amount
        
        return {
            "name": goal_name,
            "progress": progress,
            "on_track": on_track,
            "remaining_days": remaining_days,
            "projected_completion": self.project_goal_completion(goal)
        }
    
    def project_goal_completion(self, goal):
        """Project when a goal will be completed based on current progress"""
        current_amount = goal["current_amount"]
        target_amount = goal["target_amount"]
        monthly_contribution = goal.get("monthly_contribution", 0) # Use .get for safety
        
        remaining_amount = target_amount - current_amount
        if monthly_contribution <= 0:
            return "Never (need contributions)"
        
        months_remaining = remaining_amount / monthly_contribution
        completion_date = datetime.now() + timedelta(days=months_remaining*30)
        
        return completion_date.strftime("%Y-%m-%d")
    
    def recommend_portfolio(self):
        """Recommend investment portfolio based on risk profile"""
        # Get allocation based on risk profile
        allocation = risk_profile_to_allocation(self.user_data["risk_profile"])
        
        # Recommended ETFs based on allocation
        recommendations = {
            "Stocks": [
                {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation": allocation["Stocks"] * 0.7},
                {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF", "allocation": allocation["Stocks"] * 0.3}
            ],
            "Bonds": [
                {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "allocation": allocation["Bonds"] * 0.7},
                {"ticker": "BNDX", "name": "Vanguard Total International Bond ETF", "allocation": allocation["Bonds"] * 0.3}
            ],
            "Cash": [
                {"ticker": "VMFXX", "name": "Vanguard Federal Money Market Fund", "allocation": allocation["Cash"]}
            ],
            "Alternative": [
                {"ticker": "VNQ", "name": "Vanguard Real Estate ETF", "allocation": allocation["Alternative"] * 0.5},
                {"ticker": "IAU", "name": "iShares Gold Trust", "allocation": allocation["Alternative"] * 0.5}
            ]
        }
        
        return {
            "allocation": allocation,
            "recommendations": recommendations
        }
    
    def simulate_investment_returns(self, initial_amount, monthly_contribution, years, risk_profile=None):
        """Simulate investment returns based on risk profile"""
        if not risk_profile:
            risk_profile = self.user_data["risk_profile"]
        
        # Expected returns and volatility by risk profile
        risk_returns = {
            "very_conservative": {"return": 0.04, "volatility": 0.05},
            "conservative": {"return": 0.05, "volatility": 0.08},
            "moderate": {"return": 0.07, "volatility": 0.12},
            "aggressive": {"return": 0.09, "volatility": 0.16},
            "very_aggressive": {"return": 0.10, "volatility": 0.20}
        }
        
        profile_data = risk_returns.get(risk_profile, risk_returns["moderate"])
        expected_return = profile_data["return"]
        volatility = profile_data["volatility"]
        
        # Generate monte carlo simulations (simplified)
        num_simulations = 1000
        num_months = int(years * 12)
        
        # Create array to store simulation results
        results = np.zeros((num_simulations, num_months))
        
        for sim in range(num_simulations):
            portfolio_value = initial_amount
            for month in range(num_months):
                # Random monthly return based on expected return and volatility
                monthly_return = np.random.normal(
                    expected_return / 12, 
                    volatility / np.sqrt(12)
                )
                
                # Add monthly contribution
                portfolio_value += monthly_contribution
                
                # Apply return
                portfolio_value *= (1 + monthly_return)
                
                # Store result
                results[sim, month] = portfolio_value
        
        # Calculate percentiles for confidence intervals
        percentiles = {
            "10th": np.percentile(results[:, -1], 10),
            "25th": np.percentile(results[:, -1], 25),
            "50th": np.percentile(results[:, -1], 50),
            "75th": np.percentile(results[:, -1], 75),
            "90th": np.percentile(results[:, -1], 90)
        }
        
        # For visualization, take a few sample paths
        sample_paths = results[np.random.choice(num_simulations, 10, replace=False), :]
        
        return {
            "percentiles": percentiles,
            "sample_paths": sample_paths,
            "months": list(range(1, num_months + 1))
        }
    
    def suggest_debt_payoff_strategy(self):
        """Suggest debt payoff strategy (snowball vs. avalanche)"""
        debts = self.user_data["personal_info"].get("debt", {})
        
        # Dummy interest rates for demonstration
        interest_rates = {
            "credit_card": 0.18,
            "student_loans": 0.045,
            "car_loan": 0.06,
            "mortgage": 0.035,
            "personal_loan": 0.10
        }
        
        # Create list of debts with details
        debt_list = []
        for debt_name, amount in debts.items():
            # Handle cases where interest rate might not be known or is 0
            rate = interest_rates.get(debt_name, 0.05) 
            debt_list.append({
                "name": debt_name,
                "amount": float(amount), # Ensure amount is float for sorting
                "interest_rate": float(rate)
            })
        
        # Ensure debt_list is always returned, even if empty
        if not debt_list:
            return {"debt_list": [], "snowball": [], "avalanche": [], "recommendation": "No debts to analyze!"}

        # Snowball method (smallest to largest)
        snowball = sorted(debt_list, key=lambda x: x["amount"])
        
        # Avalanche method (highest interest to lowest)
        avalanche = sorted(debt_list, key=lambda x: x["interest_rate"], reverse=True)
        
        # Simple recommendation based on total interest if sum of products > 1000
        # Otherwise, snowball for psychological win.
        total_interest_cost = sum(debt["amount"] * debt["interest_rate"] for debt in debt_list)
        recommendation = "avalanche" if total_interest_cost > 1000 else "snowball"

        return {
            "debt_list": debt_list,
            "snowball": snowball,
            "avalanche": avalanche,
            "recommendation": recommendation
        }
        
    def ask_financial_question(self, question):
        """Use LLM to answer financial questions"""
        global llm
        if llm is None:
            try:
                # IMPORTANT CHANGE: Use BedrockChat for Claude 3 models
                llm = BedrockChat(
                    client=client, 
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0" 
                )
            except Exception as e:
                st.error(f"Error initializing LLM for financial question: {e}")
                return "I'm sorry, I cannot answer financial questions at this moment due to an LLM initialization error."

        if self.knowledge_base is None:
            self.knowledge_base = setup_financial_knowledge_base()
        
        # If still None, fallback to Claude directly
        if self.knowledge_base is None:
            full_prompt = f"""
            User profile:
            - Age: {self.user_data['personal_info'].get('age', 'unknown')}
            - Income: ${self.user_data['personal_info'].get('income_yearly', 0):,.2f}
            - Net Worth: ${self.calculate_net_worth():,.2f}
            - Risk Profile: {self.user_data['risk_profile']}
            
            Answer this question as a financial coach: {question}
            """
            response = query_claude(full_prompt)
            self.user_data["chat_history"].append({"question": question, "answer": response})
            return response

        # Otherwise, use the vector-based QA system
        enriched_question = f"""
        User Profile:
        - Age: {self.user_data['personal_info'].get('age', 'unknown')}
        - Income: ${self.user_data['personal_info'].get('income_yearly', 0):,.2f}/year
        - Risk Profile: {self.user_data['risk_profile']}
        - Net Worth: ${self.calculate_net_worth():,.2f}

        Based on this user profile, please answer the following question:
        {question}
        """
        result = self.knowledge_base({"question": enriched_question, "chat_history": []})
        self.user_data["chat_history"].append({"question": question, "answer": result["answer"]})
        return result["answer"]
def try_yfinance_general_news():
    """Try to get general market news using yfinance"""
    try:
        import yfinance as yf
        
        # Try to get news from major market indices or popular tickers
        market_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        
        all_articles = []
        
        for ticker in market_tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                ticker_news = ticker_obj.news
                
                if ticker_news:
                    # Filter out articles with no title or "No title"
                    for article in ticker_news:
                        title = article.get('title', '').strip()
                        if title and title != 'No title' and len(title) > 5:
                            all_articles.append(article)
                
                # Break after getting some articles to avoid too many API calls
                if len(all_articles) >= 10:
                    break
                    
            except Exception as e:
                continue  # Skip failed tickers
        
        return all_articles if all_articles else []
        
    except ImportError:
        return []
    except Exception as e:
        return []

def try_yahoo_rss_feeds():
    """Try multiple Yahoo Finance RSS feeds"""
    try:
        import feedparser
        from datetime import datetime
        
        rss_feeds = [
            ("https://feeds.finance.yahoo.com/rss/2.0/headline", "Yahoo Finance Headlines"),
            ("https://finance.yahoo.com/rss/topstories", "Yahoo Finance Top Stories"),
            ("https://feeds.finance.yahoo.com/rss/2.0/topstories", "Yahoo Finance News")
        ]
        
        all_articles = []
        
        for feed_url, source_name in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                if feed.entries:
                    for entry in feed.entries[:6]:  # Limit per feed
                        article = {
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'link': entry.link,
                            'publisher': source_name,
                            'date': getattr(entry, 'published', datetime.now().strftime('%Y-%m-%d %H:%M')),
                            'source': 'RSS'
                        }
                        all_articles.append(article)
            except Exception:
                continue
        
        return all_articles
        
    except ImportError:
        return []
    except Exception:
        return []

def try_yahoo_scraping():
    """Scrape Yahoo Finance latest news as fallback"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
        from datetime import datetime
        
        # Yahoo Finance latest news URL
        url = "https://finance.yahoo.com/news/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Look for news articles in various possible containers
        article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'(story|article|news)', re.I))
        
        if not article_elements:
            article_elements = soup.find_all('li', class_=re.compile(r'(stream|story|article)', re.I))
        
        if not article_elements:
            article_elements = soup.find_all('div', attrs={'data-test-locator': re.compile(r'(story|article)', re.I)})
        
        if not article_elements:
            article_elements = soup.find_all('div', class_=re.compile(r'(Mb\(5px\)|Pos\(r\)|Ov\(h\))', re.I))
        
        for element in article_elements[:8]:  # Limit to first 8 found elements
            try:
                # Try to extract title
                title_elem = element.find(['h3', 'h2', 'h4', 'a'], class_=re.compile(r'(title|headline)', re.I))
                if not title_elem:
                    title_elem = element.find(['a', 'h3', 'h2', 'h4'])
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                # Try to extract link
                link_elem = title_elem if title_elem.name == 'a' else element.find('a')
                link = ""
                if link_elem and link_elem.get('href'):
                    href = link_elem.get('href')
                    if href.startswith('/'):
                        link = f"https://finance.yahoo.com{href}"
                    elif href.startswith('http'):
                        link = href
                    else:
                        link = f"https://finance.yahoo.com/news/{href}"
                
                # Try to extract summary/description
                summary_elem = element.find(['p', 'div'], class_=re.compile(r'(summary|description|excerpt)', re.I))
                if not summary_elem:
                    summary_elem = element.find('p')
                
                summary = ""
                if summary_elem:
                    summary = summary_elem.get_text(strip=True)
                    if len(summary) > 300:
                        summary = summary[:300] + "..."
                
                # Try to extract time/date
                time_elem = element.find(['time', 'span'], class_=re.compile(r'(time|date)', re.I))
                pub_time = ""
                if time_elem:
                    pub_time = time_elem.get_text(strip=True)
                
                # Create article object
                article = {
                    'title': title,
                    'summary': summary,
                    'link': link or "https://finance.yahoo.com/news/",
                    'publisher': 'Yahoo Finance (Scraped)',
                    'date': pub_time or datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'source': 'Scraping'
                }
                
                articles.append(article)
                
            except Exception as e:
                continue  # Skip problematic articles
        
        return articles
        
    except Exception as e:
        return []

def try_marketwatch_rss():
    """Try MarketWatch RSS feed"""
    try:
        import feedparser
        from datetime import datetime
        
        feed_url = "http://feeds.marketwatch.com/marketwatch/topstories/"
        feed = feedparser.parse(feed_url)
        
        articles = []
        if feed.entries:
            for entry in feed.entries[:6]:
                article = {
                    'title': entry.title,
                    'summary': getattr(entry, 'summary', ''),
                    'link': entry.link,
                    'publisher': 'MarketWatch',
                    'date': getattr(entry, 'published', datetime.now().strftime('%Y-%m-%d %H:%M')),
                    'source': 'RSS'
                }
                articles.append(article)
        
        return articles
        
    except Exception:
        return []

def try_cnbc_rss():
    """Try CNBC RSS feed"""
    try:
        import feedparser
        from datetime import datetime
        
        feed_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        feed = feedparser.parse(feed_url)
        
        articles = []
        if feed.entries:
            for entry in feed.entries[:6]:
                article = {
                    'title': entry.title,
                    'summary': getattr(entry, 'summary', ''),
                    'link': entry.link,
                    'publisher': 'CNBC',
                    'date': getattr(entry, 'published', datetime.now().strftime('%Y-%m-%d %H:%M')),
                    'source': 'RSS'
                }
                articles.append(article)
        
        return articles
        
    except Exception:
        return []

def get_sample_news():
    """Generate sample news as fallback"""
    from datetime import datetime
    
    return [
        {
            "title": "Market Update: S&P 500 Reaches New Highs",
            "summary": "The S&P 500 index continues its upward trajectory amid positive economic indicators.",
            "link": "https://finance.yahoo.com",
            "publisher": "Market News Sample",
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "source": "Sample"
        },
        {
            "title": "Federal Reserve Maintains Interest Rates",
            "summary": "The Fed decided to keep interest rates steady in their latest meeting.",
            "link": "https://finance.yahoo.com",
            "publisher": "Financial Times Sample",
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "source": "Sample"
        },
        {
            "title": "Tech Stocks Lead Market Rally",
            "summary": "Technology stocks are driving today's market gains with strong earnings reports.",
            "link": "https://finance.yahoo.com", 
            "publisher": "Tech News Sample",
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "source": "Sample"
        }
    ]

def display_recent_news(agent):
    st.title("Recent Market News")
    
    # Get user's portfolio tickers
    portfolio_assets = agent.user_data.get("investment_portfolio", {}).get("assets", [])
    tickers = [asset["ticker"] for asset in portfolio_assets if "ticker" in asset]
    
    if tickers:
        st.info(f"Showing news for your portfolio: {', '.join(tickers)}")
        # Display portfolio news first
        try_ticker_news(tickers)
    else:
        st.info("No tickers in portfolio. Showing general market news.")
    
    st.subheader("📰 Latest Market News from Multiple Sources")
    
    # Collect news from multiple sources
    all_news_sources = []
    
    # Source 1: Yahoo Finance via yfinance
    yf_articles = try_yfinance_general_news()
    if yf_articles:
        all_news_sources.append(("Yahoo Finance (yfinance)", yf_articles[:6]))
    
    # Source 2: Yahoo Finance RSS feeds
    rss_articles = try_yahoo_rss_feeds()
    if rss_articles:
        all_news_sources.append(("Yahoo Finance RSS", rss_articles[:6]))
    
    # Source 3: MarketWatch RSS
    mw_articles = try_marketwatch_rss()
    if mw_articles:
        all_news_sources.append(("MarketWatch", mw_articles[:6]))
    
    # Source 4: CNBC RSS
    cnbc_articles = try_cnbc_rss()
    if cnbc_articles:
        all_news_sources.append(("CNBC", cnbc_articles[:6]))
    
    # Source 5: Yahoo Finance scraping
    scraped_articles = try_yahoo_scraping()
    if scraped_articles:
        all_news_sources.append(("Yahoo Finance (Web)", scraped_articles[:6]))
    
    # Display news from up to 5 sources
    sources_displayed = 0
    for source_name, articles in all_news_sources:
        if sources_displayed >= 5:  # Limit to 5 sources
            break
            
        if articles:  # Only display if we have articles
            st.subheader(f"🌐 {source_name}")
            
            # Remove duplicates by title within this source
            seen_titles = set()
            unique_articles = []
            
            for article in articles:
                title = article.get('title', '').strip()
                title_clean = title.lower().replace(' ', '').replace('.', '').replace(',', '')
                
                if title_clean not in seen_titles and len(title) > 10:
                    seen_titles.add(title_clean)
                    unique_articles.append(article)
            
            # Display articles from this source
            for article in unique_articles[:4]:  # Max 4 articles per source
                display_article_multi_source(article)
            
            sources_displayed += 1
            st.markdown("---")  # Separator between sources
    
    # If no sources worked, show sample news
    if sources_displayed == 0:
        st.info("Live news feeds temporarily unavailable - showing sample news")
        sample_articles = get_sample_news()
        all_news_sources.append(("Sample News", sample_articles))
        
        st.subheader("📰 Sample Market News")
        for article in sample_articles:
            display_article_multi_source(article)
    
    # Show summary of sources
    if sources_displayed > 0:
        st.success(f"✅ Displaying news from {sources_displayed} different sources")
    
    # Refresh button
    if st.button("🔄 Refresh All News Sources"):
        st.rerun()

def display_article_multi_source(article, show_large=False):
    """Helper function to display a single article from any source"""
    try:
        from datetime import datetime
        
        # Skip articles with no title or invalid titles
        title = article.get('title', '').strip()
        if not title or title == 'No title' or len(title) <= 5:
            return
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Check if it's from yfinance (has thumbnail structure)
            if 'thumbnail' in article:
                thumbnail = article.get('thumbnail', {})
                if thumbnail and 'resolutions' in thumbnail:
                    try:
                        img_url = thumbnail['resolutions'][-1]['url']
                        st.image(img_url, width=120)
                    except:
                        st.write("📰")
                else:
                    st.write("📰")
            else:
                # For RSS feeds and other sources
                st.write("📈")
        
        with col2:
            st.markdown(f"**{title}**")
            
            # Publisher and date info
            publisher = article.get('publisher', 'Unknown')
            
            # Handle different date formats
            if 'providerPublishTime' in article:
                # yfinance format
                pub_time = article.get('providerPublishTime', 0)
                if pub_time:
                    pub_date = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                    st.caption(f"{publisher} • {pub_date}")
                else:
                    st.caption(publisher)
            else:
                # RSS and other formats
                date_str = article.get('date', '')
                st.caption(f"{publisher} • {date_str}")
            
            # Summary if available
            summary = article.get('summary', '')
            if summary:
                # Clean HTML tags for RSS feeds
                import re
                clean_summary = re.sub('<[^<]+?>', '', summary)
                st.write(clean_summary[:200] + "..." if len(clean_summary) > 200 else clean_summary)
            
            # Link to full article
            if 'link' in article and article['link']:
                st.markdown(f"[Read more]({article['link']})")
            
            st.markdown("")  # Small spacing
            
    except Exception as e:
        pass  # Skip problematic articles silently

def try_ticker_news(tickers):
    """Try to get news for specific tickers"""
    try:
        import yfinance as yf
        
        articles_displayed = False
        portfolio_news_shown = False
        
        for ticker in tickers[:5]:  # Limit to first 5 tickers
            try:
                ticker_obj = yf.Ticker(ticker)
                ticker_news = ticker_obj.news
                
                # Filter out articles with no title or "No title"
                valid_news = []
                if ticker_news:
                    for article in ticker_news:
                        title = article.get('title', '').strip()
                        if title and title != 'No title' and len(title) > 5:
                            valid_news.append(article)
                
                if valid_news:  # Only display if valid news exists
                    if not portfolio_news_shown:
                        st.subheader("📈 News for Your Portfolio")
                        portfolio_news_shown = True
                    
                    articles_displayed = True
                    st.markdown(f"### {ticker.upper()} News")
                    
                    # Display first 3 valid articles for each ticker
                    for article in valid_news[:3]:
                        display_article_multi_source(article)
                    
                    st.markdown("")  # Add space between tickers
                    
            except Exception as e:
                continue  # Silently skip tickers that fail
        
        return articles_displayed
        
    except ImportError:
        return False
    except Exception as e:
        return False

def init_streamlit():
    st.set_page_config(
        page_title="Personal Finance & Investment Coach",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check if agent exists in session state, if not create it
    if "financial_agent" not in st.session_state:
        st.session_state.financial_agent = FinancialAgent()
    
# Moved ALL display functions OUTSIDE of init_streamlit()
def display_financial_goals(agent):
    st.title("Financial Goals")
    st.subheader("Track and Manage Your Goals")
    
    # Add new goal section
    with st.expander("➕ Add New Goal"):
        col1, col2 = st.columns(2)
        with col1:
            goal_name = st.text_input("Goal Name", key="new_goal_name")
            target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=100.0, key="new_goal_target_amount")
        with col2:
            current_amount = st.number_input("Current Amount ($)", min_value=0.0, step=100.0, key="new_goal_current_amount")
            target_date = st.date_input("Target Date", key="new_goal_target_date")
            
        if st.button("Add Goal", key="add_new_goal_btn"):
            if goal_name and target_amount > 0:
                new_goal = {
                    "name": goal_name,
                    "target_amount": target_amount,
                    "current_amount": current_amount,
                    "target_date": target_date.isoformat(),
                    "monthly_contribution": 0 # Add a default monthly contribution for new goals
                }
                if "goals" not in agent.user_data:
                    agent.user_data["goals"] = []
                agent.user_data["goals"].append(new_goal)
                st.success(f"Goal '{goal_name}' added successfully!")
                st.rerun()
            else:
                st.warning("Please enter a valid goal name and target amount.")
    
    # Display existing goals
    goals = agent.user_data.get("goals", [])
    if goals:
        for i, goal in enumerate(goals):
            progress = agent.track_goal_progress(goal["name"])
            if progress:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{goal['name']}**")
                    progress_value = min(progress["progress"] / 100, 1.0)
                    st.progress(progress_value)
                    st.write(f"${goal['current_amount']:,} / ${goal['target_amount']:,}")
                
                with col2:
                    st.write(f"Progress: {progress['progress']:.1f}%")
                    if 'projected_completion' in progress:
                        st.write(f"Est. completion: {progress['projected_completion']}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_goal_{i}"):
                        agent.user_data["goals"].pop(i)
                        st.success("Goal deleted!")
                        st.rerun()
                
                st.write("---")
    else:
        st.info("No financial goals set yet. Add your first goal above!")

    st.markdown("---")
    if st.button("Clear Goals Data", key="clear_goals_data"):
        agent.clear_goals()
        st.rerun()

# --- Streamlit UI ---
def main():
    init_streamlit()
    agent = st.session_state.financial_agent
    agent.load_user_data() # Ensure data is loaded on every run, important after clears

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Goals", "Budget", "Investments", "Debt Management", "Financial Q&A", "Upload Statement", "Market News"])

    if page == "Dashboard":
        st.title("💰 Personal Finance Dashboard")
        st.write(f"Welcome back! Let's review your financial health.")

        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Net Worth", f"${agent.calculate_net_worth():,.2f}")
        with col2:
            st.metric("Monthly Income", f"${agent.get_monthly_income():,.2f}")
        with col3:
            total_monthly_expenses = sum(agent.get_monthly_expenses_by_category().values())
            st.metric("Monthly Expenses", f"${total_monthly_expenses:,.2f}")

        st.subheader("Quick Insights")
        st.info(f"Your current risk profile is: **{agent.user_data['risk_profile'].replace('_', ' ').title()}**.")
        
        # Display goals summary
        goals = agent.user_data.get("goals", [])
        if goals:
            st.write("### Goal Progress Summary")
            for goal in goals:
                progress = agent.track_goal_progress(goal["name"])
                if progress:
                    st.progress(min(progress["progress"] / 100, 1.0), text=f"{goal['name']}: {progress['progress']:.1f}%")

    elif page == "Goals":
        display_financial_goals(agent)

    elif page == "Budget":
        st.title("💸 Budget & Spending Analysis")
        st.subheader("Monthly Spending by Category")

        expenses_by_category = agent.get_monthly_expenses_by_category()
        if expenses_by_category:
            expenses_df = pd.DataFrame(list(expenses_by_category.items()), columns=["Category", "Amount"])
            fig = px.pie(expenses_df, values='Amount', names='Category', title='Average Monthly Spending by Category')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Raw Expenses Data")
            # Display raw data, but ensure categories are cleaned for consistency
            raw_expenses_df = agent.user_data["expenses"].copy()
            if not raw_expenses_df.empty:
                raw_expenses_df["category"] = raw_expenses_df["category"].apply(clean_category)
            st.dataframe(raw_expenses_df)
        else:
            st.info("No expense data available to display budget. Upload a statement or add manual transactions.")

        st.subheader("Spending Trends Over Time")
        monthly_trends = agent.analyze_spending_trends()
        if not monthly_trends.empty:
            fig_trends = px.line(monthly_trends, x="month", y="amount", color="category", title="Monthly Spending Trends")
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No spending trends to display.")
        
        st.markdown("---")
        if st.button("Clear Expenses Data", key="clear_expenses_data"):
            agent.clear_expenses()
            st.rerun()
        if st.button("Clear Income Data", key="clear_income_data"):
            agent.clear_income()
            st.rerun()

    # Investment Tab
    if page == "Investments":
        st.title("📈 Investment Portfolio")
        st.subheader("Current Portfolio Value")
        # Fetch and display the total portfolio value using the agent's method
        st.metric("Total Investment Value", f"${agent.get_investment_portfolio_value():,.2f}")

        st.subheader("Current Holdings")
        if agent.user_data["investment_portfolio"].get("assets"):
            holdings_df = pd.DataFrame(agent.user_data["investment_portfolio"]["assets"])
            st.dataframe(holdings_df)
        else:
            st.info("No investments currently held. Add some below!")

        st.subheader("Add New Investment")
        with st.form("new_investment_form"):
            ticker = st.text_input("Ticker Symbol (e.g., AAPL, SPY)").upper()
            name = st.text_input("Company/Fund Name")
            shares = st.number_input("Number of Shares", min_value=0.01, format="%.2f")
            purchase_price = st.number_input("Purchase Price Per Share", min_value=0.01, format="%.2f")
            asset_type = st.selectbox("Asset Type", ["ETF", "Stock", "Bond", "Mutual Fund", "Other"])
            
            submitted = st.form_submit_button("Add Investment")
            if submitted:
                if ticker and name and shares > 0 and purchase_price > 0:
                    agent.add_investment(ticker, name, shares, purchase_price, asset_type)
                    # Force a rerun to update the displayed portfolio after adding an investment
                    st.rerun() 
                else:
                    st.error("Please fill in all investment details correctly.")

        st.subheader("Asset Allocation Recommendation")
        recommended_portfolio = agent.recommend_portfolio()
        st.write("Based on your **" + agent.user_data["risk_profile"].replace('_', ' ') + "** risk profile, here's a recommended allocation:")
        allocation_df = pd.DataFrame(list(recommended_portfolio["allocation"].items()), columns=["Asset Class", "Percentage"])
        fig_alloc = px.pie(allocation_df, values='Percentage', names='Asset Class', title='Recommended Asset Allocation')
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.subheader("Recommended ETFs")
        for asset_class, etfs in recommended_portfolio["recommendations"].items():
            st.write(f"**{asset_class} ({recommended_portfolio['allocation'][asset_class] * 100:.1f}%):**")
            for etf in etfs:
                st.write(f"- {etf['name']} ({etf['ticker']}) - {(etf['allocation'] * 100):.1f}% of total portfolio")

        # Start of Monte Carlo Simulation section
        st.subheader("Investment Growth Simulation")
        initial_invest = st.number_input("Initial Investment Amount ($)", min_value=100.0, 
                                            value=max(100.0, float(agent.user_data["investment_portfolio"].get("cash", 5000))), 
                                            step=100.0)
        monthly_cont = st.number_input("Monthly Contribution ($)", min_value=0.0, value=500.0, step=50.0)
        years_sim = st.slider("Simulation Years", min_value=1, max_value=40, value=20)
        
        simulation_results = agent.simulate_investment_returns(initial_invest, monthly_cont, years_sim, agent.user_data["risk_profile"])
        
        st.write(f"Projected portfolio value after {years_sim} years:")
        for percentile, value in simulation_results["percentiles"].items():
            st.write(f"- {percentile} percentile: **${value:,.2f}**")
        
        fig_sim = go.Figure()
        for path in simulation_results["sample_paths"]:
            fig_sim.add_trace(go.Scatter(y=path, mode='lines', name='Sample Path', opacity=0.3))
        
        fig_sim.add_trace(go.Scatter(y=[simulation_results["percentiles"]["50th"]] * len(simulation_results["months"]), # Corrected this line
                                     mode='lines', name='Median', line=dict(dash='dash', color='red')))
        fig_sim.update_layout(title='Monte Carlo Investment Simulation',
                                xaxis_title='Months', yaxis_title='Portfolio Value ($)')
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.markdown("---")
        if st.button("Clear Investment Data", key="clear_investment_data"):
            agent.clear_investment_portfolio()
            st.rerun()
        # End of Monte Carlo Simulation section

    elif page == "Debt Management":
        st.title("📉 Debt Management")
        st.subheader("Your Current Debts")
        
        debts = agent.user_data["personal_info"].get("debt", {})
        if debts:
            debt_df = pd.DataFrame([{"Debt Type": k, "Amount": v} for k, v in debts.items()])
            st.dataframe(debt_df)
            st.metric("Total Debt", f"${sum(debts.values()):,.2f}")
        else:
            st.info("You currently have no recorded debts. Great job!")

        st.subheader("Add New Debt Manually")
        with st.expander("➕ Add New Debt"):
            new_debt_name = st.text_input("Debt Name (e.g., Student Loan, Car Loan)", key="new_debt_name")
            new_debt_amount = st.number_input("Debt Amount ($)", min_value=0.0, step=100.0, key="new_debt_amount")
            new_debt_interest_rate = st.number_input("Annual Interest Rate (e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="new_debt_interest_rate")
            
            if st.button("Add Debt", key="add_debt_btn"):
                if new_debt_name and new_debt_amount > 0:
                    # Ensure personal_info and debt dictionary exist
                    if "personal_info" not in agent.user_data:
                        agent.user_data["personal_info"] = {}
                    if "debt" not in agent.user_data["personal_info"]:
                        agent.user_data["personal_info"]["debt"] = {}

                    agent.user_data["personal_info"]["debt"][new_debt_name] = new_debt_amount
                    st.success(f"Debt '{new_debt_name}' of ${new_debt_amount:,.2f} added.")
                    st.rerun()
                else:
                    st.warning("Please enter a valid debt name and amount.")

        st.subheader("Debt Payoff Strategies")
        strategy_info = agent.suggest_debt_payoff_strategy()
        
        # Corrected: Check if 'debt_list' exists and is not empty
        if "debt_list" in strategy_info and strategy_info["debt_list"]:
            st.write("Consider these strategies for paying off your debts:")
            
            st.markdown("#### Debt Snowball Method")
            st.write("Pay off debts from smallest balance to largest. Once the smallest is paid, roll that payment into the next smallest.")
            snowball_df = pd.DataFrame(strategy_info["snowball"])
            st.dataframe(snowball_df.style.format({"amount": "${:,.2f}", "interest_rate": "{:.2%}"}))
            
            st.markdown("#### Debt Avalanche Method")
            st.write("Pay off debts from highest interest rate to lowest. This method saves you the most money on interest.")
            avalanche_df = pd.DataFrame(strategy_info["avalanche"])
            st.dataframe(avalanche_df.style.format({"amount": "${:,.2f}", "interest_rate": "{:.2%}"}))
            
            st.markdown(f"**Our Recommendation: The {strategy_info['recommendation'].title()} Method** (based on estimated interest savings/psychological wins).")
        else:
            st.info("No debts to analyze for payoff strategies.")
        
        st.markdown("---")
        if st.button("Clear Debt Data", key="clear_debt_data"):
            agent.clear_personal_info_debt()
            st.rerun()


    elif page == "Financial Q&A":
        st.title("💬 Ask Your Financial Coach")
        st.write("Ask any financial question, and I'll do my best to answer based on general knowledge and your profile.")

        user_question = st.text_input("Your Question:")
        if st.button("Ask"):
            if user_question:
                with st.spinner("Thinking..."):
                    response = agent.ask_financial_question(user_question)
                    st.write("---")
                    st.write(response)
            else:
                st.warning("Please enter a question.")
        
        st.subheader("Chat History")
        if agent.user_data["chat_history"]:
            for chat in reversed(agent.user_data["chat_history"]): # Show most recent first
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**Coach:** {chat['answer']}")
                st.markdown("---")
        else:
            st.info("No conversation history yet.")

    elif page == "Upload Statement":
        st.title("⬆️ Upload Credit Card Statement")
        st.write("Upload your credit card statement (PDF) to analyze your spending.")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            st.write("Processing your statement...")
            extracted_transactions_df = parse_credit_card_statement(uploaded_file)
            
            if not extracted_transactions_df.empty:
                st.subheader("Extracted Transactions")
                st.dataframe(extracted_transactions_df)
                
                # Option to append to existing expenses
                if st.button("Add these transactions to my expenses"):
                    if isinstance(agent.user_data["expenses"], pd.DataFrame) and not agent.user_data["expenses"].empty:
                        # Ensure concatenation also applies cleaning if the old data wasn't cleaned
                        agent.user_data["expenses"] = pd.concat([agent.user_data["expenses"], extracted_transactions_df], ignore_index=True)
                        agent.user_data["expenses"]["category"] = agent.user_data["expenses"]["category"].apply(clean_category) # Re-clean all categories
                    else:
                        agent.user_data["expenses"] = extracted_transactions_df
                    st.success("Transactions added to your financial data!")
                    st.rerun() # Rerun to update dashboard/budget pages
        
        st.markdown("---")
        st.subheader("Global Data Management")
        if st.button("Clear All User Data", key="clear_all_data_btn"):
            agent.clear_all_data()
            st.rerun() # Rerun to reflect cleared data

    elif page == "Market News":
        display_recent_news(agent)
        # Add refresh button here after the news display
        # With this:
        if st.button("🔄 Refresh News", key="refresh_news_main"):
            st.rerun()

if __name__ == "__main__":
    main()