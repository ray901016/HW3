import numpy as np
import plotly.graph_objs as go
from sklearn.svm import LinearSVC

def generate_data(threshold):
    np.random.seed(0)
    num_points = 600
    mean = 0
    variance_x1 = 20  # x1 方向的方差
    variance_x2 = 5   # x2 方向的方差

    # 生成橢圓形分布數據集 centered at (0, 0)
    x1 = np.random.normal(mean, np.sqrt(variance_x1), num_points)
    x2 = np.random.normal(mean, np.sqrt(variance_x2), num_points)
    
    # 定義標籤 Y，根據動態閾值設定橢圓形邊界
    distances = (x1 / np.sqrt(variance_x1))**2 + (x2 / np.sqrt(variance_x2))**2
    Y = np.where(distances < threshold, 0, 1)  # 動態閾值由滑桿設定

    # 計算 x3 使用高斯函數來生成第三維
    def gaussian_function(x1, x2):
        return np.exp(-0.1 * (x1**2 + x2**2))
    
    x3 = gaussian_function(x1, x2)

    # 創建網格來顯示平面
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    return np.column_stack((x1, x2)), Y, x3, xx, yy  # 只返回5個變量

def train_svm(X, Y, x3, xx, yy):
    clf = LinearSVC(random_state=0, max_iter=10000, dual=False)
    clf.fit(X, Y)
    coef = clf.coef_[0]
    intercept = clf.intercept_

    # 根據紅點的最高點動態設置分隔平面高度
    red_points_max_height = np.max(x3[Y == 1])  # 計算紅點的最高點
    zz = np.full_like(xx, red_points_max_height)  # 動態平面高度基於紅點的最高點

    # 繪製數據點和分隔平面
    scatter = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=x3,
        mode='markers',
        marker=dict(
            size=5,
            color=['blue' if label == 0 else 'red' for label in Y],
            opacity=0.7
        ),
        name='Data points'
    )

    plane = go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
        opacity=0.5,
        showscale=False,
        name='Separating Hyperplane'
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='x1'),
            yaxis=dict(title='x2'),
            zaxis=dict(title='x3')
        ),
        title="3D Scatter Plot with Y Color and Separating Hyperplane",
        showlegend=True
    )

    return go.Figure(data=[scatter, plane], layout=layout)
