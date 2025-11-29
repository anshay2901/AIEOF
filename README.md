# AIEOF
# ⚡ AI Industrial Energy Forecaster  
Viksit Bharat 2047 – Environmental Sustainability Challenge (Tech-Driven Track)

Live Demo 👉 **https://aieof-toi.streamlit.app/**

This project builds an AI-powered system to forecast India's industrial electricity demand, detect anomalies, analyze hourly peaks, and offer load optimization suggestions using Prophet, synthetic MoSPI-aligned datasets, and a modern Streamlit dashboard.

---

## 🚀 Features
- Daily & hourly electricity forecasting  
- Peak load forecasting (next 30 days)  
- **Model accuracy**  
  - MAE ≈ **107,333 MWh**  
  - RMSE ≈ **118,550 MWh**  
  - MAPE ≈ **5.87%**  
- Anomaly detection using Isolation Forest  
- Hourly heatmap  
- Peak-shaving optimization  
- CSV export panel  
- Fully interactive Streamlit dashboard  

---

## 📺 Live Dashboard  
**https://aieof-toi.streamlit.app/**  
Hosted for free on Streamlit Cloud.

---

## 📁 Project Structure
AIEOF/
│
├── app/ # Streamlit dashboard
├── models/ # Exported forecast CSVs
├── data/
│ ├── raw/ # Original datasets (NOT committed)
│ └── processed/ # Cleaned/synthetic datasets
├── notebooks/ # Jupyter notebooks (01 & 02)
├── .gitignore
└── README.md


---

## 🚀 Features
- Daily & hourly electricity forecasting  
- Peak load forecasting (next 30 days)  
- **Model accuracy**
  - MAE ≈ **107,333 MWh**
  - RMSE ≈ **118,550 MWh**
  - MAPE ≈ **5.87%**
- Anomaly detection using Isolation Forest  
- Hourly profile heatmap  
- Peak-shaving optimization (10% demo)  
- CSV export panel  
- Fully interactive Streamlit dashboard  

---

## 🛠️ Installation & Setup

### 1️⃣ Create virtual environment  
python -m venv .venv

### 2️⃣ Activate it  
**Windows PowerShell**
..venv\Scripts\Activate.ps1

### 3️⃣ Install dependencies  
pip install -r requirements.txt

### 4️⃣ Run the Streamlit app  
streamlit run app/app.py

---

## 📦 Tech Stack
- Python  
- Prophet  
- Pandas / NumPy  
- scikit-learn  
- Streamlit  
- Matplotlib / Plotly  

---

## 📝 Notes
- Raw MoSPI Excel files (`data/raw/`) **NOT be committed to Git**.  
- Synthetic processed datasets **are safe to store** (`data/processed/`).  
- Models regenerate anytime using the notebooks.

---

## 🙌 Author
Built by **Anshay Singh**  
for the **Viksit Bharat 2047 – Environmental Sustainability Challenge**.
