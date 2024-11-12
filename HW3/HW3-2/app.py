import streamlit as st
import plotly.graph_objs as go
from utils.data_generation import generate_data, train_svm

st.title("3D SVM Visualization on Circular Dataset")
st.write("This application visualizes a 2D dataset distributed in a circular pattern with a separating hyperplane created by an SVM in 3D.")

# Generate data and train SVM model
X, Y, x3, xx, yy, zz = generate_data()
fig = train_svm(X, Y, x3, xx, yy, zz)

st.plotly_chart(fig)
