# Disaster Response Pipeline Project

### Overview
This project is a web app which classifies an emergency message into several categories which can be useful to route the messages to the appropriate specialized teams.

### Requirements
- Following python libraries : pandas, nltk, flask, plotly, sklearn, sqlalchemy, pickle

### Description
The root contains three folders :
- 'app' : contains the web application related files. Use the run.py file in this folder to initiate the web app.
- 'data' : contains
   - data files
   - 'process_data.py' : contains ETL pipeline
- 'models' : contains 'train_classifier.py' which acts as ML pipeline

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Open http://0.0.0.0:3000/

### Website Images
1. Overview
<img src="/resources/overview">
2. Message search response
<img src="/resources/response">
