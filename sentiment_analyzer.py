from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "I love this movie",
    "This is amazing",
    "I really enjoyed this",
    "What a wonderful experience",
    "I hate this movie",
    "This is terrible",
    "I really disliked this",
    "What a horrible experience"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

print(X.toarray())

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X, labels)

new_text = input("Enter a sentence: ")

new_text_vectorized = vectorizer.transform([new_text])

prediction = model.predict(new_text_vectorized)

print("Sentiment:", prediction[0])
