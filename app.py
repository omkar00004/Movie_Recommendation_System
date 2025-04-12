import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Load movies data
movies = pickle.load(open('movies.pkl', 'rb'))  # rb means read binary
movies['title'] = movies['title'].apply(lambda x: x.title())  # make it into title case
movies_list = movies['title'].values

# Load similarity matrix parts and reconstruct
@st.cache_resource  # Cache this to avoid reloading every time
def load_similarity():
    # Check if the original similarity file exists
    if os.path.exists('similarity.pkl'):
        return pickle.load(open('similarity.pkl', 'rb'))
    
    # Check if split files exist
    split_files_exist = all(os.path.exists(f'similarity_part{i}.pkl') for i in range(1, 5))
    
    if not split_files_exist:
        st.error("Similarity matrix files not found. Please make sure they exist in the correct location.")
        st.info("If you have the original similarity.pkl file, you need to split it first.")
        
        # Provide a way to split the file if it exists
        if os.path.exists('similarity.pkl') and st.button("Split similarity matrix"):
            with st.spinner("Splitting similarity matrix..."):
                # Load the original matrix
                similarity = pickle.load(open('similarity.pkl', 'rb'))
                
                # Get the shape
                shape = similarity.shape
                half_row = shape[0] // 2
                half_col = shape[1] // 2
                
                # Split the matrix
                part1 = similarity[:half_row, :half_col]
                part2 = similarity[:half_row, half_col:]
                part3 = similarity[half_row:, :half_col]
                part4 = similarity[half_row:, half_col:]
                
                # Save each part
                pickle.dump(part1, open('similarity_part1.pkl', 'wb'))
                pickle.dump(part2, open('similarity_part2.pkl', 'wb'))
                pickle.dump(part3, open('similarity_part3.pkl', 'wb'))
                pickle.dump(part4, open('similarity_part4.pkl', 'wb'))
                
                st.success("Split complete! Refresh the page to load the split files.")
                
        return np.zeros((1, 1))  # Return empty matrix as placeholder
    
    # Load the split files
    part1 = pickle.load(open('similarity_part1.pkl', 'rb'))
    part2 = pickle.load(open('similarity_part2.pkl', 'rb'))
    part3 = pickle.load(open('similarity_part3.pkl', 'rb'))
    part4 = pickle.load(open('similarity_part4.pkl', 'rb'))
    
    # Reconstruct the matrix
    top_half = np.hstack((part1, part2))
    bottom_half = np.hstack((part3, part4))
    return np.vstack((top_half, bottom_half))

# Try to load similarity matrix
try:
    similarity = load_similarity()
    similarity_loaded = True
except Exception as e:
    st.error(f"Error loading similarity matrix: {e}")
    similarity_loaded = False

# Load posters data from CSV
posters_df = pd.read_csv('movies_tmdb_posters.csv')
# Create a dictionary for quick poster lookup
poster_dict = dict(zip(posters_df['tmdb_id'], posters_df['poster']))

def get_poster(movie_id):
    """Get poster URL from the poster dictionary"""
    if movie_id in poster_dict:
        return poster_dict[movie_id]
    else:
        # Return a placeholder image if poster not found
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

def recommend(movie):
    # Convert titles to lowercase for comparison
    movies['title'] = movies['title'].apply(lambda x: x.lower())
    movie_index = movies[movies['title'] == movie.lower()].index[0]
    distances = similarity[movie_index]
    # Get more recommendations than needed in case some don't have posters
    movies_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:15]

    recommended_movies = []
    recommended_movies_poster = []
    
    # Loop through recommendations until we have 5 movies with posters
    for i in movies_indices:
        if len(recommended_movies) >= 5:
            break
            
        movie_id = movies.iloc[i[0]].movie_id
        
        # Only add the movie if it has a poster
        if movie_id in poster_dict:
            recommended_movies.append(movies.iloc[i[0]].title.title())
            recommended_movies_poster.append(poster_dict[movie_id])
    
    # If we still don't have 5 movies, pad with placeholders
    while len(recommended_movies) < 5:
        recommended_movies.append("No more recommendations")
        recommended_movies_poster.append("https://via.placeholder.com/500x750?text=No+More+Recommendations")
            
    return recommended_movies, recommended_movies_poster

st.title("Movie Recommendation System")

selected_movie_name = st.selectbox(
    "Select a movie you like",  # Updated the label text to be more descriptive
    (movies_list),
)

if st.button("Recommend"):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])