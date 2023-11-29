import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Streamlit app
st.title("Breast Cancer Prediction App (Decision Tree)")

# Add a sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    features = {}
    for i, feature_name in enumerate(data.feature_names):
        min_val = float(data.data[:, i].min())
        max_val = float(data.data[:, i].max())
        mean_val = float(data.data[:, i].mean())
        features[feature_name] = st.sidebar.slider(f'{feature_name} ({min_val:.2f} - {max_val:.2f})', min_val, max_val, mean_val)
    return list(features.values())

# Add a slider for adjusting the max_depth parameter
max_depth = st.sidebar.slider("Select Max Depth", min_value=1, max_value=20, value=5)

user_input = user_input_features()

# Train a Decision Tree classifier with the specified max_depth
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
prediction = clf.predict([user_input])

# Plot scatter plot before and after adjusting depth
fig_before = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train, labels={'x': 'Feature 0', 'y': 'Feature 1'},
                        title='Scatter Plot Before Depth Adjustment')
fig_after = px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=clf.predict(X_train),
                       labels={'x': 'Feature 0', 'y': 'Feature 1'}, title=f'Scatter Plot After Depth Adjustment (Max Depth: {max_depth})')

# Display plots
st.plotly_chart(fig_before)
st.plotly_chart(fig_after)

# Display prediction results
st.subheader('Class Labels and their corresponding index number')
st.write(data.target_names)

st.subheader('Prediction')
st.write(data.target_names[prediction[0]])

st.subheader('Prediction Probability')
st.write(clf.predict_proba([user_input]))

# Additional Information about the dataset
st.sidebar.subheader('Dataset Information')
st.sidebar.text('Number of classes: 2')
st.sidebar.text('Number of features: {}'.format(len(data.feature_names)))
st.sidebar.text('Number of samples: {}'.format(len(data.data)))


