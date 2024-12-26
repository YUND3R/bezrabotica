import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os
from google.colab import files

# Загрузка данных
file_path = '/content/drive/My Drive/Mashinka/data.csv'
df = pd.read_csv(file_path)

# Далее продолжаем обработку данных, как в вашем коде:
columns_to_keep = [
    'territory',
    'num_economactivepopulation_all',
    'employed_num_all',
    'unemployed_num_all',
    'eactivity_lvl',
    'employment_lvl',
    'unemployment_lvl',
    'dis_unagegroup_30-39',
    'dis_emagegroup_30-39',
    'num_unagegroup_30-39',
    'num_emagegroup_30-39',
    'year'
]
df = df[columns_to_keep]

df.fillna(df.mean(numeric_only=True), inplace=True)  # Уточнено использование numeric_only

def clean_territory(value):
    if isinstance(value, str) and len(value) > 100:
        return value[:100]
    return value

df['territory'] = df['territory'].apply(clean_territory)
df['territory'] = df['territory'].astype(str).str.strip()
territory_mapping = {territory: idx for idx, territory in enumerate(df['territory'].unique())}
df['territory'] = df['territory'].map(territory_mapping)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.drop(['year'], axis=1))
normalized_df = pd.DataFrame(normalized_data, columns=[col for col in df.columns if col != 'year'])
normalized_df['year'] = df['year'].values

# Преобразуем данные для использования в LSTM
sequence_length = 10  # Длина временной последовательности

# Создаем массивы для входных данных и меток
X = []
y = []

for i in range(len(normalized_df) - sequence_length):
    X.append(normalized_df.iloc[i:i+sequence_length, :-1].values)
    y.append(normalized_df.iloc[i + sequence_length, -2])  # Пример: прогнозируем предпоследний столбец

X = np.array(X)
y = np.array(y)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализуем метки
label_scaler = MinMaxScaler()
y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Создаем улучшенную модель LSTM
model = Sequential()

# Первый слой LSTM с большим количеством юнитов
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.4))  # Увеличиваем вероятность отбора
model.add(BatchNormalization())

# Второй слой LSTM
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.4))  # Увеличиваем вероятность отбора
model.add(BatchNormalization())

# Третий слой LSTM
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.4))  # Увеличиваем вероятность отбора
model.add(BatchNormalization())

# Полносвязный слой
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Выходной слой
model.add(Dense(1, activation='linear'))

# Компилируем модель
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])



# Callback для обучения
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Мониторинг на валидационном наборе
    patience=10,  # Увеличиваем patience до 10 эпох
    restore_best_weights=True
)

# Callback для уменьшения скорости обучения
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,  # Увеличиваем patience для ReduceLROnPlateau
    min_lr=1e-5
)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=100,  # Количество эпох остаётся 100
    batch_size=32,  # Размер батча
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

# Оцениваем модель
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {loss}, Test MAE: {mae}")

# Пример прогноза
predictions = model.predict(X_test)
predictions = label_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
print("Пример прогнозов:", [round(pred, 2) for pred in predictions[:5]])
