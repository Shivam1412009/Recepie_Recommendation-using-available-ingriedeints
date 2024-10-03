import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import time
from streamlit_lottie import st_lottie
import json




tfidf = joblib.load(r'tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load(r'tfidf_matrix.pkl')
df = joblib.load(r'recipe_dataframe.pkl')

def recommend_recipes(user_input, df, tfidf, tfidf_matrix):
    user_tfidf = tfidf.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    recommended_indices = cosine_similarities.argsort()[-5:][::-1]
    return df.iloc[recommended_indices]


st.title('🍽️ Recipe Recommendation 🥘')




st.markdown("""Welcome to the Indian Food Recipe Recommendation System! Get ready to cook something amazing!""")

filepath="hero_section_animation.json"
def load_lottiefiles(filepath:str):
    with open(filepath,"r",encoding="utf-8") as f:
        return json.load(f)

st_lottie(load_lottiefiles(filepath),width=400,height=300,loop=True,key="loading",speed=1)



user_input = st.text_input("Enter ingredients or food items you have:")

if st.button('Recommend Recipes'):
    if user_input:  
        recommendations = recommend_recipes(user_input, df, tfidf, tfidf_matrix)
        with st.spinner(text='Recommending top reciepes'): #**! Displating a progress bar before recommendation
            time.sleep(3)
        st.write("✨ Top recommended recipes for you:")
        for index, row in recommendations.iterrows():
            st.subheader(f"🍲 {row['name']}")
            st.write(f"**Ingredients:** {row['ingredients']}")
            st.write(f"**Prep_instruction:** {row['prep_instruction']}")
            ##! have to work on the images :(
            st.write(f"🔗 [Image]({row['image_url']})")
            
      
    else:
        st.write("🚫 Please enter some ingredients or food items to get started.")

