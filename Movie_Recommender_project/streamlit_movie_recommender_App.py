import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# this dataset was downloaded from https://data.world/data-society/imdb-5000-movie-dataset
movies_dts = pd.read_csv("movie_dataset.csv")
features = ['keywords','cast','genres','director']

st.title('Recommending that is most relevant to You!')
st.subheader(f"Enter name of the Movie you like most, to see top similar movies like that one | 'Please use a suitable name i.e. [Avatar, Gravity, Wanted, Spider-Man]'")
st.write("## Note: Our site will recommend you movies that are released on or before 2016")
st.sidebar.title('Movie Recommender')

def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

for i in features:
    # filling all NaNs with blank string
    movies_dts[i] = movies_dts[i].fillna('')
#
#applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column
movies_dts["combined_features"] = movies_dts.apply(combine_features,axis=1)
a = movies_dts.iloc[0].combined_features

cv = CountVectorizer()
count_matrix = cv.fit_transform(movies_dts["combined_features"]) #feeding combined strings(movie contents) to CountVectorizer() object

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return movies_dts[movies_dts.index == index]["title"].values[0]
def get_index_from_title(title):
    return movies_dts[movies_dts.title == title]["index"].values[0]

movie_user_likes = st.text_input("Write Movie name and hit Enter")
try:
    movie_index = get_index_from_title(movie_user_likes)
    # accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    i=0
    st.write("## Top 10 similar movies to "+movie_user_likes+" are:\n")
    for element in sorted_similar_movies:
        st.write(get_title_from_index(element[0]))
        i+=1
        if i>10:
           break
except:
    st.write("Please enter a valid movie name.")


st.sidebar.header("About Recommendation Models")
st.sidebar.info("Recommendation systems are used to give users a better experience of what is most relevant to them.")
st.sidebar.header("About App")
st.sidebar.info("A Simple ML-Streamlit App for Movies Recommendation ")
st.sidebar.text("Maintained by Farhan Ali and Team")


