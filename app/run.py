import json
import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/MessCat.db')
df = pd.read_sql_table('MessCat', engine)

# load model
model = joblib.load("../models/MessCat_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categ =df.drop(['id','message','original','genre'], axis=1)

    data=[]

    for columns in categ:
        data.append(categ[columns].sum())

    categ_list =list(df.drop(['id','message','original','genre'], axis=1))
    categ_sums =pd.Series(data=data, index=categ_list)

    df_categ_sums=pd.DataFrame(categ_sums)

    df_categ_sums.reset_index(inplace=True)

    df_categ_sums=df_categ_sums.rename(columns={'index':'Categories'})

    df_categ_sums=df_categ_sums.rename(columns={0:'Count'})

    df_categ_high=df_categ_sums[df_categ_sums.Count > df_categ_sums.Count.mean()]

    df_categ_high.sort_values('Count', ascending=False, inplace=True)

    df_categ_low=df_categ_sums[df_categ_sums.Count < df_categ_sums.Count.mean()]

    df_categ_low.sort_values('Count', ascending=False, inplace=True)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_categ_high.Categories,
                    y=df_categ_high.Count
                )
            ],

            'layout': {
                'title': 'Higher than average Count of Messages of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_categ_low.Categories,
                    y=df_categ_low.Count
                )
            ],

            'layout': {
                'title': 'Lower than Average Count of Messages of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
