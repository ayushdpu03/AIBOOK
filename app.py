from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient, errors
import certifi
import os

app = Flask(__name__, static_url_path='/static')

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/library?retryWrites=true&w=majority&tls=true")

try:
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client['library']
    feedback_collection = db['feedback']
    print("Connected to MongoDB")
except errors.ServerSelectionTimeoutError as e:
    print(f"MongoDB connection error: {e}")
    feedback_collection = None

# Read CSV file (Load a subset of data to save memory)
try:
    new_df = pd.read_csv("Final_ai.csv", usecols=['books', 'img', 'mod_title', 'rating'], nrows=1000)
except FileNotFoundError as e:
    print(f"Error: {e}")
    new_df = pd.DataFrame()  # Use an empty dataframe if file is not found

# Initialize vectorizer with limited features
cv = CountVectorizer(max_features=1000, stop_words="english")
vectors = cv.fit_transform(new_df['books']).toarray()
similar = cosine_similarity(vectors)

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

@app.route('/')
def home():
    try:
        data = new_df[new_df['rating'] == 5].sort_values(by="rating", ascending=False).head(8)
    except KeyError as e:
        print(f"Error: {e}")
        data = pd.DataFrame()

    return render_template('home.html', total_data=data.to_dict('records'),
                           author_data=data['books'].tolist(),
                           image_data=data['img'].tolist(),
                           title_data=data['mod_title'].tolist(),
                           rating_data=data['rating'].tolist(),
                           genre_data=data['mod_title'].tolist())

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = []
    error = False
    if request.method == 'POST':
        title_input = request.form.get('title_input', 'None').strip()

        def recommend_fun(book):
            recommended_books = []
            try:
                book_index = new_df[new_df['mod_title'] == book].index[0]
                distances = similar[book_index]
                book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

                for i in book_list:
                    recommended_books.append([
                        new_df.iloc[i[0]].mod_title,
                        new_df.iloc[i[0]].img,
                        new_df.iloc[i[0]].rating,
                        new_df.iloc[i[0]].books
                    ])
                return recommended_books

            except (IndexError, KeyError) as e:
                print(f"Exception occurred: {e}")
                return None

        data = recommend_fun(title_input)

        if data is None:
            error = True

    return render_template('recommend.html', data=data, error=error)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        if feedback_collection is None:
            return render_template('feedback.html', error="Database connection error. Please try again later.")
        
        try:
            feedback_data = {
                'title': request.form.get('title', '').strip(),
                'author': request.form.get('author', '').strip(),
                'genre': request.form.get('genre', '').strip(),
                'img_url': request.form.get('img-url', '').strip(),
                'rating': request.form.get('rating', '').strip()
            }

            # Ensure all fields are provided
            if not all(feedback_data.values()):
                raise ValueError("All fields are required")

            feedback_collection.insert_one(feedback_data)
            print("Feedback successfully inserted")
            return redirect(url_for('home'))
        
        except errors.PyMongoError as e:
            print(f"PyMongo error: {e}")
            return render_template('feedback.html', error="Database error. Please try again later.")
        except ValueError as e:
            print(f"Value error: {e}")
            return render_template('feedback.html', error=str(e))
        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('feedback.html', error="An unexpected error occurred. Please try again later.")

    return render_template('feedback.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
