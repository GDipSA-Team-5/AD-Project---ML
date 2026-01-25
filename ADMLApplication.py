
print("Hello World!")
print("This is the AD ML Application.")

# -------------------------------------------------------------------------------------------------------------------
# Example: A trained scikit-learn model
from flask import Flask, request, jsonify
import pickle
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

model = svm.SVC()
model.fit(X, y)

filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# -------------------------------------------------------------------------------------------------------------------
# Flask API example (Must listen on 0.0.0.0)
app = Flask(__name__)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# health check api test endpoint
@app.route("/health")
def health():
        return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    pred = model.predict([data["features"]])
    return jsonify({"prediction": pred.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

# Install Docker: https://docs.docker.com/get-started/get-docker/
# Before deploying to Azure — always test the docker container locally using the following commands:
# docker build -t ad-ml-flask .
# docker run -p 8000:8000 ad-ml-flask
# curl http://localhost:8000/health
# --- Test Method 1 ---
"""
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1, 2, 3, 4]} 
"""
# --- Test Method 2 ---
"""
Invoke-RestMethod `
  -Uri http://localhost:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"features":[1,2,3,4]}'
"""

