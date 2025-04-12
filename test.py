import pickle
import streamlit as st
import pandas as pd

movies_poster = pd.read_csv('movies_tmdb_posters.csv')  # or your actual file path

def fetch_poster(tmdb_id):
    movie = movies_poster[movies_poster['tmdb_id'] == tmdb_id]
    if not movie.empty:
        return movie.iloc[0]['poster']
    return "https://via.placeholder.com/500x750?text=No+Image"


def recommend(movie):
    index = movies_poster[movies_poster['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        tmdb_id = movies_poster.iloc[i[0]]['tmdb_id']
        poster_url = fetch_poster(tmdb_id)
        recommended_movie_names.append(movies_poster.iloc[i[0]]['title'])
        recommended_movie_posters.append(poster_url)

    return recommended_movie_names, recommended_movie_posters



st.header('🎬 Movie Recommender System')

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i])
