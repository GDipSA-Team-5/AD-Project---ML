# AD-Project---ML
Team 5 GDipSA - AD Project ML Component
<br>Uses Flask API and Python Pickle module
<br>Pickle is commonly used with a Flask API to serialize Python objects (such as trained machine learning models) into a byte stream for storage in a file, and then deserialize them when the API is run to make predictions.

### Install Docker: https://docs.docker.com/get-started/get-docker/
### Before deploying to Azure — always test the docker container locally using the following commands:
- docker build -t ad-ml-flask .
- docker run -p 8000:8000 ad-ml-flask
- curl http://localhost:8000/health
<br> --- Test Method 1 ---
<br> curl -X POST http://localhost:8000/predict \
<br>  -H "Content-Type: application/json" \
<br>  -d '{"features":[1, 2, 3, 4]}
 --- Test Method 2 ---
<br> Invoke-RestMethod `
<br>  -Uri http://localhost:8000/predict `
<br>  -Method POST `
<br>  -ContentType "application/json" `
<br>  -Body '{"features":[1,2,3,4]}'