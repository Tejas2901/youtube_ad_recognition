import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr


df = pd.read_csv('youtube_sub_cleaned.csv')

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Subtitle'])

# Create a recognizer object
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak your search query:")
    audio = recognizer.listen(source)

# Convert the audio to text 
try:
    user_input = recognizer.recognize_google(audio)
    print(f"You said: {user_input}")
except sr.UnknownValueError:
    print("Sorry, I couldn't understand what you said.")
    user_input = ""

user_input_vector = tfidf_vectorizer.transform([user_input])
cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)

# Find the most similar index
most_similar_index = cosine_similarities.argmax()


# Get the corresponding target value
most_similar_target = df.at[most_similar_index, 'Title']
print(f"The most similar target value is: {most_similar_target}")
