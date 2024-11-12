import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 資料生成：建立 300 個隨機樣本 X(i)，範圍在 0 到 1000 之間
np.random.seed(42)
X = np.random.randint(0, 1001, 300).reshape(-1, 1)

# 二元標籤：當 500 < X(i) < 800 時，Y(i)=1，否則 Y(i)=0
Y = np.where((X > 500) & (X < 800), 1, 0)

# 訓練測試分割：將資料分為 80% 的訓練集和 20% 的測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 模型訓練
# 邏輯回歸模型
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_logreg = logreg.predict(X_test)
logreg_prob = logreg.predict_proba(np.linspace(0, 1000, 300).reshape(-1, 1))[:, 1]

# 支持向量機 (SVM) 模型
svm = SVC(probability=True)
svm.fit(X_train, Y_train)
y_pred_svm = svm.predict(X_test)
svm_prob = svm.predict_proba(np.linspace(0, 1000, 300).reshape(-1, 1))[:, 1]

# 視覺化
plt.figure(figsize=(15, 6))

# 子圖 1：邏輯回歸預測結果及概率邊界
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='True Labels')  # 原始數據點，灰色表示真實標籤
plt.scatter(X_test, y_pred_logreg, color='orange', marker='x', label='Logistic Regression Prediction')  # 邏輯回歸預測結果，橘色
plt.plot(np.linspace(0, 1000, 300), logreg_prob, color='orange', linestyle='--', label='Logistic Regression Boundary')  # 邏輯回歸虛線決策邊界
plt.xlabel('X')
plt.ylabel('Y / Probability')
plt.title('X vs Y and Logistic Regression Prediction')
plt.legend()

# 子圖 2：SVM 預測結果及概率邊界
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='True Labels')  # 原始數據點，灰色表示真實標籤
plt.scatter(X_test, y_pred_svm, color='purple', marker='s', label='SVM Prediction')  # SVM 預測結果，紫色
plt.plot(np.linspace(0, 1000, 300), svm_prob, color='purple', linestyle='--', label='SVM Boundary')  # SVM 虛線決策邊界
plt.xlabel('X')
plt.ylabel('Y / Probability')
plt.title('X vs Y and SVM Prediction')
plt.legend()

plt.tight_layout()
plt.show()
