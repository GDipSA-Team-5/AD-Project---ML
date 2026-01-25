# base image
FROM python:3.9-slim

# working dir
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# expose port your app listens on
EXPOSE 8000

# start command — adapt if you’re using Flask/uvicorn
CMD ["uvicorn", "ADMLApplication:app", "--host", "0.0.0.0", "--port", "8000"]
