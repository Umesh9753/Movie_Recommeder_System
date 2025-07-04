import streamlit as st
import pandas as pd
import pickle
import requests

# --- Optional: Function to fetch movie poster (API key needed) ---
def fetch_poster(movie_id):
    api_key = 'fcd3a842'  # Replace with your TMDB API key
    url = f"http://www.omdbapi.com/?i=tt3896198&apikey=fcd3a842{movie_id}?api_key={fcd3a842}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"

# --- Load model and data ---
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# --- Recommend function ---
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# --- Streamlit UI ---
st.title('Movie Recommender System')

select_movie_name = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(select_movie_name)
    for i in recommendations:
        st.write(i)

import numpy as np
np.savez_compressed("similarity_compressed.npz", similarity=similarity)

# Load
data = np.load("similarity_compressed.npz")
similarity = data['similarity']
