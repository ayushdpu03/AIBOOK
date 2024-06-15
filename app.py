from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import certifi

# Initialize Flask application
app = Flask(__name__, static_url_path='/static')

# Load the dataset
new_df = pd.read_csv("Final_ai.csv")

# MongoDB configuration
client = MongoClient(
    "mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/library?retryWrites=true&w=majority&tls=true&tlsCAFile="
    + certifi.where()
)
db = client.library
books_collection = db.books_data
feedback_collection = db.feedback

@app.route('/')
def home():
    # Retrieve top-rated books from MongoDB
    data = list(books_collection.find({"rating": 5}).sort("rating", 1).limit(8))
    author_names = [row['books'] for row in data]
    genre = [row['mod_title'] for row in data]
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
        title_input = request.form.get('title_input', '')

        # CountVectorizer for text processing
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(new_df['books']).toarray()

        # Compute cosine similarity matrix
        similar = cosine_similarity(vectors)

        # Stemming function
        ps = PorterStemmer()
        def stem(text):
            return " ".join([ps.stem(word) for word in text.split()])

        # Recommendation function
        def recommend_fun(book):
            recommended_books = []
            try:
                book_index = new_df[new_df['mod_title'] == book].index[0]
                distances = similar[book_index]
                book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

                for i in book_list:
                    recommended_books.append({
                        'title': new_df.iloc[i[0]].mod_title,
                        'img': new_df.iloc[i[0]].img,
                        'rating': new_df.iloc[i[0]].rating,
                        'books': new_df.iloc[i[0]].books
                    })
                return recommended_books

            except IndexError as e:
                print(f"Exception occurred: {e}")
                return []

        data = recommend_fun(title_input)
        if not data:
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
        
        # Insert feedback into MongoDB
        feedback_collection.insert_one({
            'title': title,
            'author': author,
            'genre': genre,
            'img_url': img_url,
            'rating': rating
        })
        
        print("Feedback recorded successfully")
        return redirect(url_for('home'))

    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
