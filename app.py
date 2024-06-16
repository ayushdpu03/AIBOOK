from flask import Flask, render_template, request, jsonify, redirect, url_for
import pymongo
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Load your CSV data
new_df = pd.read_csv("Final_ai.csv")

app = Flask(__name__)

# MongoDB configuration
client = pymongo.MongoClient("mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/library?retryWrites=true&w=majority&appName=Cluster0")
db = client.library

@app.route('/')
def home():
    books_data = db.books_data.find({"rating": 5}).sort("rating", pymongo.ASCENDING).limit(8)
    data = list(books_data)
    author_names = [row['author'] for row in data]
    genre = [row['genre'] for row in data]
    image = [row['img'] for row in data]
    rating = [row['rating'] for row in data]
    title = [row['mod_title'] for row in data]

    return render_template('home.html', total_data=data,
                           author_data=author_names,
                           image_data=image,
                           title_data=title,
                           rating_data=rating,
                           genre_data=genre)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = []
    error = False
    if request.method == 'POST':
        title_input = request.form.get('title_input', 'None')
        print(title_input)

        cv = CountVectorizer(max_features=5000, stop_words="english")
        cv.fit_transform(new_df['books']).toarray().shape
        vectors = cv.fit_transform(new_df['books']).toarray()

        similar = cosine_similarity(vectors)
        ps = PorterStemmer()

        def stem(text):
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)

        def recommend_fun(book):
            recommended_books = []
            try:
                book_index = new_df[new_df['mod_title'] == book].index[0]
                distances = similar[book_index]
                book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

                for i in book_list:
                    item = []
                    item.extend(list([new_df.iloc[i[0]].mod_title]))
                    item.extend(list([new_df.iloc[i[0]].img]))
                    item.extend(list([new_df.iloc[i[0]].rating]))
                    item.extend(list([new_df.iloc[i[0]].books]))
                    recommended_books.append(item)
                return recommended_books

            except (IndexError, KeyError) as e:
                print('\n\n', f"Exception occurred: {e}")

        data = recommend_fun(title_input)
        print('\n', "data: ", data, '\n')

        if data is None:
            error = True

    return render_template('recommend.html', data=data, error=error)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        genre = request.form['genre']
        rating = request.form['rating']
        img_url = request.form['img-url']

        feedback_data = {
            'title': title,
            'author': author,
            'genre': genre,
            'rating': rating,
            'img_url': img_url
        }
        
        db.feedback.insert_one(feedback_data)
        print("Feedback submitted successfully")
        return redirect(url_for('home'))

    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
