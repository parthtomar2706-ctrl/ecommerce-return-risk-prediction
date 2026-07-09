# End-to-End E-Commerce Product Return Risk & Production API

## 📌 Project Overview
High return rates are a major margin-killer for modern e-commerce enterprises. This project implements a production-grade machine learning system designed to predict the probability of a customer returning a purchased item at checkout. 

This repository goes beyond simple data science modeling by serving the trained pipeline as a live, low-latency **REST API via FastAPI**, converting raw intelligence into real-time business decisions.

## 🛠️ Key Technical Implementations
* **ML Pipeline:** Programmed simulated enterprise customer/order datasets, using **SMOTE** (Synthetic Minority Over-sampling Technique) inside validation splits to handle structural class imbalances.
* **Production Serialization:** Serialized the feature-scaling and classification engine into reusable pipeline artifacts (`joblib`).
* **REST API Microservice:** Built a high-performance backend microservice utilizing **FastAPI** and **Pydantic** validation schemas to accept transaction JSON payloads and instantly calculate financial overhead risks.

## 🚀 Getting Started

### 1. Installation
Clone this repository and install dependencies:
```bash
git clone [https://github.com/parthtomar2706-ctrl/ecommerce-return-risk-prediction.git](https://github.com/parthtomar2706-ctrl/ecommerce-return-risk-prediction.git)
cd ecommerce-return-risk-prediction
pip install -r requirements.txt
