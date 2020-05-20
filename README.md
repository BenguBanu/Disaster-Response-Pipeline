# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

In this project I worked on a data set containing real messages that were sent during disaster events.(provided by Figure Eight) I created a machine learning pipeline to categorize these events so that messages can be send to an appropriate disaster relief agency.

Project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

The coding for this project

app | - template | |- master.html # main page of web app | |- go.html # classification result page of web app |- run.py # Flask file that runs app

data |- disaster_categories.csv # data to process |- disaster_messages.csv # data to process |- process_data.py |- MessCat.db # database to save clean data to

models |- train_classifier.py |- MessCat_model.pkl # saved model

README.md
