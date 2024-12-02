import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, KFold # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.utils import to_categorical, plot_model # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Konfüzyon Matrisi Görselleştirme Fonksiyonu
def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    print(f"Konfüzyon matrisi '{title}.png' olarak kaydedildi.")
    plt.close()

# Veri Setini Yükleme
file_path = 'iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(file_path, header=None, names=columns)

# Özellik ve Sınıfları Ayırma
X = data.iloc[:, :-1].values  # Özellikler
y = data.iloc[:, -1].values   # Sınıf bilgisi

# Sınıfları Sayısal Hale Getirme
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Normalizasyon için scaler
scaler = StandardScaler()

# 1. Eğitim Setini Aynı Zamanda Test Verisi Olarak Kullanma
print("\n1. Egitim Setini Ayni Zamanda Test Verisi Olarak Kullanma:")
X_all = scaler.fit_transform(X)  # Tüm veri normalleştirilir
y_all = to_categorical(y)        # One-hot encoding
model = Sequential([
    Dense(8, input_dim=X_all.shape[1], activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_all, y_all, epochs=50, batch_size=10, verbose=1)
loss, accuracy = model.evaluate(X_all, y_all, verbose=0)
print(f"Doğruluk: {accuracy:.2f}")

# Modelin görsel görünümü
plot_model(model, to_file='model_topology.png', show_shapes=True, show_layer_names=True)
print("Ağın görsel görünümü 'model_topology.png' olarak kaydedildi.")

# 2. 5-Fold Cross Validation Kullanarak
print("\n2. 5-Fold Cross Validation Kullanarak:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_5fold = []
labels = encoder.classes_

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    model = Sequential([
        Dense(8, input_dim=X_train.shape[1], activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies_5fold.append(accuracy)

    # Konfüzyon Matrisi
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, f"Confusion Matrix Fold {fold+1}", labels)

print(f"5-Fold Cross Validation Dogruluk Ortalamasi: {np.mean(accuracies_5fold):.2f}")

# 3. 10-Fold Cross Validation Kullanarak
print("\n3. 10-Fold Cross Validation Kullanarak:")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies_10fold = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    model = Sequential([
        Dense(8, input_dim=X_train.shape[1], activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies_10fold.append(accuracy)

    # Konfüzyon Matrisi
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, f"Confusion Matrix Fold {fold+1} (10-Fold)", labels)

print(f"10-Fold Cross Validation Dogruluk Ortalamasi: {np.mean(accuracies_10fold):.2f}")

# 4. %66-%34 Eğitim Test Ayırarak (5 Farklı Rastgele Ayırma ile)
print("\n4. %66-%34 Egitim Test Ayirarak (5 Farkli Rastgele Ayirma ile):")
accuracies_random_split = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=i)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    model = Sequential([
        Dense(8, input_dim=X_train.shape[1], activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies_random_split.append(accuracy)

    # Konfüzyon Matrisi
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, f"Confusion Matrix Random Split {i+1}", labels)

print(f"%66-%34 Egitim/Test Ayirmasi Dogruluk Ortalamasi: {np.mean(accuracies_random_split):.2f}")