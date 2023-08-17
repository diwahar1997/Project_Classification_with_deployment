


# In[9]:
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the trained models and TF-IDF vectorizers
nb_classifier = joblib.load('nb_model.pkl')
lr_classifier = joblib.load('lr_model.pkl')
svm_classifier = joblib.load('svm_model.pkl')
rf_classifier = joblib.load('rf_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
def main():
    st.title("Article Prediction App")
    st.write("This app predicts whether an article is true or false based on its content.")

    # Model selection
    selected_model = st.radio("Select Model:", ("Naive Bayes", "Logistic Regression", "SVM", "Random Forest"))

    # Text input area for user to input article text
    user_input = st.text_area("Article Text", "")

    if st.button("Predict"):
        if user_input:
            if selected_model == "Naive Bayes":
                model = nb_classifier
            elif selected_model == "Logistic Regression":
                model = lr_classifier
            elif selected_model == "SVM":
                model = svm_classifier
            elif selected_model == "Random Forest":
                model = rf_classifier

            user_input_tfidf = tfidf_vectorizer.transform([user_input])
            user_prediction = model.predict(user_input_tfidf)
            prediction_probability = model.predict_proba(user_input_tfidf)[0][1]
            st.write("User Input:", user_input)
            st.write("Predicted Outcome:", "True" if user_prediction[0] == 1 else "False")
            st.write("Prediction Probability:", f"{prediction_probability:.2f}")

    # Display a bar graph comparing model performance
    st.write("### Model Performance Comparison")
    models = ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
    accuracy_scores = [0.85, 0.89, 0.88, 0.91]  # Example accuracy scores, replace with actual scores
    plt.bar(models, accuracy_scores)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    st.pyplot(plt)

if __name__ == "__main__":
    main()

# %%
