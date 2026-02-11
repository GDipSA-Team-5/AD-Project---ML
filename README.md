# GDipSA-61: Team 05 (In5nite) – ML Prediction Service

This repository contains the **Machine Learning (ML) microservice** for the In5nite project. 

## Purpose
The ML service predicts:
- Average daily fill growth for the next collection cycle
- Estimated number of days until a predefined fill threshold is reached

## Dataset

The ML model was trained using an open-source dataset:
- Source: https://zenodo.org/records/14988663
- Data cleaning and feature engineering performed by the team
- Cleaned dataset published on Hugging Face: https://huggingface.co/datasets/SA61team5/AD-tableC

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

## Security Controls Implemented

Static Analysis (SAST)
- SonarCloud code quality and security scanning
- Test coverage enforced via pytest + coverage.xml

Software Composition Analysis (SCA)
- Snyk dependency scanning

Container Security
- Docker image scanning using Snyk

Dynamic Analysis (DAST)
- OWASP ZAP baseline scan

API Hardening
- Input validation enforced
- Defensive error handling

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
