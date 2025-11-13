import streamlit as st
import pickle

tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("model/logistic_model.pkl", "rb"))

st.title("MailMind — Email Classification System")
st.subheader("Classify emails as Spam, Social, or Priority")

email_text = st.text_area("Enter email content:")

if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        vector = tfidf.transform([email_text])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Category: **{prediction.upper()}**")
