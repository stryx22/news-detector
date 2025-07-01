# utils/explain.py

import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

# This function creates a LIME explanation object for a given text
# It requires a trained model and vectorizer (TF-IDF)
def explain_prediction(text, model, vectorizer):
    class_names = ['Real', 'Fake']
    pipeline = make_pipeline(vectorizer, model)
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=10)
    return exp
