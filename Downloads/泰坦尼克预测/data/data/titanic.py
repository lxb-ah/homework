# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'train.csv')
        
        df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功！路径: {data_path}")
        print(f"数据集形状: {df.shape}")
        print("\n前3行数据预览:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")
        print("请检查：")
        print("1. 项目目录下是否有 data/train.csv 文件")
        print("2. 文件名是否为 train.csv（区分大小写）")
        exit()

data = load_data()
def preprocess(df):
    print("\n=== 数据预处理 ===")
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
    print("\n处理后的数据示例:")
    print(df.head(3))
    return df

processed_data = preprocess(data)
X = processed_data.drop('Survived', axis=1)
y = processed_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
print("\n=== 模型评估 ===")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name:>15} | 准确率: {accuracy:.4f}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("模型准确率比较")
plt.ylim(0.7, 0.9)
plt.subplot(1, 2, 2)
if 'Random Forest' in models:
    importances = models['Random Forest'].feature_importances_
    top_features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(5)
    
    sns.barplot(x='Importance', y='Feature', data=top_features, palette="rocket")
    plt.title("Top 5 重要特征")

plt.tight_layout()
plt.show()
sample_idx = np.random.randint(0, len(X_test))
sample = X_test.iloc[sample_idx:sample_idx+1]
print("\n🔮 随机乘客预测:")
print(f"特征值:\n{sample}")
print(f"\n真实结果: {'幸存' if y_test.iloc[sample_idx] == 1 else '遇难'}")
for name, model in models.items():
    pred = model.predict(scaler.transform(sample))[0]
    print(f"{name:>15}: {'幸存' if pred == 1 else '遇难'}")