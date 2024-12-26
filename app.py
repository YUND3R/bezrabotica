import tensorflow as tf
import numpy as np
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random

# Загрузка модели из локального файла
model = tf.keras.models.load_model('bezrabotica.keras')

# Загрузка данных из локального файла
data = pd.read_csv('data.csv')

# Оставляем только нужные столбцы
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
data = data[columns_to_keep]

# Заполнение пропущенных значений
data.fillna(data.mean(numeric_only=True), inplace=True)

# Очистка данных для столбца "territory"
def clean_territory(value):
    if isinstance(value, str) and len(value) > 100:
        return value[:100]
    return value

data['territory'] = data['territory'].apply(clean_territory)
data['territory'] = data['territory'].astype(str).str.strip()

# Маппинг территории
territory_mapping = {territory: idx for idx, territory in enumerate(data['territory'].unique())}
territory_reverse_mapping = {v: k for k, v in territory_mapping.items()}
data['territory'] = data['territory'].map(territory_mapping)

# Нормализация данных
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.drop(['year'], axis=1))
normalized_df = pd.DataFrame(normalized_data, columns=[col for col in data.columns if col != 'year'])
normalized_df['year'] = data['year'].values

# Функция предсказания

def predict_unemployment(territory):
    if territory not in territory_mapping:
        return "Неверное название территории. Пожалуйста, выберите из списка."

    predictions = []
    current_year = data['year'].max() + 1

    for i in range(5):
        sample_row = data[data['territory'] == territory_mapping[territory]].iloc[-1].copy()
        sample_row['year'] = current_year
        input_data = sample_row.drop('year').values.reshape(1, -1)

        # Нормализация входных данных
        input_normalized = scaler.transform(input_data)
        input_sequence = np.expand_dims(input_normalized, axis=0)

        # Прогноз
        prediction = model.predict(input_sequence)
        random_variation = random.uniform(-0.5, 0.5)  # Добавление случайной вариации
        adjusted_prediction = max(0, prediction[0][0] * 100 + random_variation)  # Гарантия, что результат не отрицательный
        predictions.append((current_year, f"{adjusted_prediction:.2f}"))

        # Обновление года
        current_year += 1

    return [[year, value] for year, value in predictions]

# Интерфейс Gradio
interface = gr.Interface(
    fn=predict_unemployment,
    inputs=[
        gr.Dropdown(label="Территория", choices=list(territory_mapping.keys()))
    ],
    outputs=[
        gr.Dataframe(headers=["Год", "Прогноз уровня безработицы (%)"], label="Прогноз на ближайшие 5 лет")
    ],
    title="Модель прогнозирования уровня безработицы",
    description="Выберите территорию для прогноза уровня безработицы на ближайшие 5 лет."
)

interface.launch()
