# Lead Propensity Demo

This repository contains a minimal FastAPI application that demonstrates lead
conversion scoring. Upload a CSV file of leads to train a simple logistic
regression model and view the propensity score for each lead. New leads can be
scored in real time from the application. A small JavaScript front end is
included for interacting with the API.

## Running the application

```
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser. The page lets you upload a CSV
file (the last column should be the binary label). After uploading the data you
can see the calculated propensity scores and add new leads to score them in
real time.
