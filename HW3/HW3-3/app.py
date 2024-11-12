import streamlit as st
import plotly.graph_objs as go
from utils.data_generation import generate_data, train_svm

# Streamlit 標題和滑桿
st.title("3D Scatter Plot with Separating Hyperplane")
st.write("This application visualizes a 2D dataset with a dynamically moving separating plane controlled by the red points' highest point.")

# 使用滑桿調整 distance threshold，範圍從 0.1 到 10.0
threshold = st.slider("Distance Threshold", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

# 生成數據並訓練 SVM 模型
X, Y, x3, xx, yy = generate_data(threshold)
fig = train_svm(X, Y, x3, xx, yy)  # 刪除 threshold 參數

# 顯示圖表並保留視角
scene_camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))  # 默認視角
fig.update_layout(scene_camera=scene_camera, uirevision='constant')  # 使用 uirevision 保留視角
st.plotly_chart(fig)
