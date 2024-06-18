from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient
import os

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'supersecretkey'  # Needed for flash messages

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/library?retryWrites=true&w=majority&appName=Cluster0")
client = MongoClient(MONGO_URI)
db = client['library']
feedback_collection = db['feedback']

# Function to load CSV in chunks
def load_csv(filename):
    chunk_size = 10000  # Adjust chunk size as needed
    csv_chunks = pd.read_csv(filename, chunksize=chunk_size)
    return pd.concat(csv_chunks, ignore_index=True)

# Read CSV file
try:
    new_df = load_csv("Final_ai.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    new_df = pd.DataFrame()  # Use an empty dataframe if file is not found

# Initialize vectorizer and compute vectors
cv = CountVectorizer(max_features=100, stop_words="english")
if not new_df.empty:
    vectors = cv.fit_transform(new_df['books']).toarray()
    similar = cosine_similarity(vectors)

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

@app.route('/')
def home():
    # Simulate MongoDB data retrieval with CSV data for the home route
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
        title = request.form.get('title')
        author = request.form.get('author')
        genre = request.form.get('genre')
        rating = request.form.get('rating')
        img_url = request.form.get('img-url')

        if not title or not author or not genre or not rating:
            flash('All fields except Image URL are required!', 'error')
            return redirect(url_for('feedback'))

        try:
            feedback_collection.insert_one({
                'title': title,
                'author': author,
                'genre': genre,
                'rating': float(rating),
                'img_url': img_url
            })
            flash('Feedback submitted successfully!', 'success')
        except Exception as e:
            flash(f'An error occurred: {e}', 'error')

        return redirect(url_for('home'))

    return render_template('feedback.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
