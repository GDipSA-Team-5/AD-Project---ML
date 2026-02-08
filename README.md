# GDipSA-61: Team 05 (In5nite) – ML Prediction Service

This repository contains the **Machine Learning (ML) microservice** for the In5nite project.  
The service provides **bin fill growth prediction and threshold estimation** via a RESTful API and is designed to be deployed as a **containerised service**.

The ML service is treated as a **black-box predictor** and is integrated with the .NET backend, which consumes the prediction results for operational decision-making.

## Purpose
The ML service predicts:
- Average daily fill growth for the next collection cycle
- Estimated number of days until a predefined fill threshold is reached

This enables **predictive waste collection planning** without embedding ML logic into the backend or mobile application.

## Dataset

The ML model is trained using an open-source dataset (https://zenodo.org/records/14988663) that was cleaned and pre-processed.

- Original dataset sourced from an open-source repository
- Data cleaning and feature engineering performed by the team
- Cleaned dataset uploaded to Hugging Face 

## Architecture Overview
- **Framework**: Flask (Python)
- **Model Loading**: `joblib`
- **Input/Output**: JSON over HTTP (REST API)
- **Deployment**: Docker → Azure Container Registry → Azure Web App for Containers
- **Integration**: Consumed by .NET backend via REST API

The ML service:
- Has **no direct database access**
- Stores **no system-critical secrets or credentials**
- Is isolated as an independent microservice

## Health Check Endpoint

### GET /health
This endpoint is used to verify that the ML service is running and reachable.  

```bash
curl http://localhost:8000/health
```

Response: 200 OK

---

## Prediction Endpoint

### POST /predict
Accepts collection cycle data and returns:
- Predicted next-cycle average daily fill growth
- Estimated number of days until threshold is reached

This endpoint is consumed by the .NET backend.

## AI Tool Declaration

ChatGPT-5.2 was used to assist with parts of the data cleaning process and documentation.

We are responsible for the content and quality of the submitted work.
