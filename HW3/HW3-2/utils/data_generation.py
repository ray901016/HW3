import numpy as np
import plotly.graph_objs as go
from sklearn.svm import LinearSVC

def generate_data():
    np.random.seed(0)
    num_points = 600
    mean, variance = 0, 10

    # 生成數據集 centered at C1=(0,0) 和 C2=(10,10)
    c1_x = np.random.normal(mean, np.sqrt(variance), num_points)
    c1_y = np.random.normal(mean, np.sqrt(variance), num_points)
    distances_c1 = np.sqrt(c1_x**2 + c1_y**2)
    Y_c1 = np.where(distances_c1 < 6, 0, 1)

    c2_x = np.random.normal(10, np.sqrt(variance), num_points)
    c2_y = np.random.normal(10, np.sqrt(variance), num_points)
    distances_c2 = np.sqrt((c2_x - 10)**2 + (c2_y - 10)**2)
    Y_c2 = np.where(distances_c2 < 3, 0, 1)

    x1 = np.concatenate((c1_x, c2_x))
    x2 = np.concatenate((c1_y, c2_y))
    Y = np.concatenate((Y_c1, Y_c2))

    # 計算 x3 使用高斯函數
    def gaussian_function(x1, x2):
        return np.exp(-0.1 * (x1**2 + x2**2))
    
    x3 = gaussian_function(x1, x2)

    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    return np.column_stack((x1, x2)), Y, x3, xx, yy, None

def train_svm(X, Y, x3, xx, yy, zz):
    clf = LinearSVC(random_state=0, max_iter=10000)
    clf.fit(X, Y)
    coef = clf.coef_[0]
    intercept = clf.intercept_

    # 使用 x3 作為第三維度，不再計算 z 值
    zz = np.full_like(xx, np.mean(x3))  # 使用 x3 的平均值表示平面高度

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
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
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
        title="3D Scatter Plot with Separating Hyperplane",
        showlegend=True
    )

    return go.Figure(data=[scatter, plane], layout=layout)
