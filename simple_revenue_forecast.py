#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая программа прогнозирования revenue_total с помощью линейной регрессии
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleRevenueForecaster:
    def __init__(self, csv_file=None):
        """Инициализация простого прогнозировщика revenue"""
        self.csv_file = csv_file
        self.df = None
        self.forecast_df = None
        self.model = None
        
    def load_data(self, csv_file=None):
        """Загрузка данных из CSV файла"""
        if csv_file:
            self.csv_file = csv_file
            
        if not self.csv_file:
            raise ValueError("Не указан файл для загрузки")
            
        print(f"Загрузка данных из {self.csv_file}...")
        # Пробуем разные разделители
        try:
            self.df = pd.read_csv(self.csv_file, sep=',')
            print(f"Загружено с разделителем ',': {len(self.df)} записей, {len(self.df.columns)} колонок")
        except:
            try:
                self.df = pd.read_csv(self.csv_file, sep=';')
                print(f"Загружено с разделителем ';': {len(self.df)} записей, {len(self.df.columns)} колонок")
            except:
                self.df = pd.read_csv(self.csv_file)
                print(f"Загружено с автоматическим определением: {len(self.df)} записей, {len(self.df.columns)} колонок")
        
        return self.df
    
    def clean_data(self):
        """Очистка и подготовка данных"""
        print("Очистка данных...")
        
        # Очистка временных колонок
        for col in ['year', 'month']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Удаляем строки с пустыми временными данными
        self.df = self.df.dropna(subset=['year', 'month'])
        
        # Очистка revenue_total
        if 'revenue_total' in self.df.columns:
            # Заменяем запятые на пустую строку и конвертируем в числа
            self.df['revenue_total'] = self.df['revenue_total'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['revenue_total'] = pd.to_numeric(self.df['revenue_total'], errors='coerce')
            # Заменяем NaN на 0
            self.df['revenue_total'] = self.df['revenue_total'].fillna(0)
        
        print(f"После очистки: {len(self.df)} записей")
        
    def prepare_time_features(self):
        """Подготовка временных признаков"""
        print("Подготовка временных признаков...")
        
        # Создаем временной индекс
        self.df['time_index'] = (self.df['year'] - self.df['year'].min()) * 12 + (self.df['month'] - 1)
        
        # Сезонные признаки
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            
            # Квартальные dummy переменные
            self.df['quarter'] = ((self.df['month'] - 1) // 3) + 1
            for q in range(1, 5):
                self.df[f'q{q}'] = (self.df['quarter'] == q).astype(int)
        
        # Праздничные периоды
        if 'month' in self.df.columns:
            self.df['holiday_period'] = (
                (self.df['month'] == 12) |  # Декабрь
                (self.df['month'] == 1) |   # Январь
                (self.df['month'] == 2) |   # Февраль
                (self.df['month'] == 3) |   # Март
                (self.df['month'] == 5)     # Май
            ).astype(int)
        
        print("Временные признаки подготовлены")
    
    def train_revenue_model(self, test_size=0.3):
        """Обучение модели для revenue_total"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ REVENUE_TOTAL")
        print("="*60)
        
        # Подготавливаем данные
        self.prepare_time_features()
        
        # Базовые временные признаки
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
        
        print(f"Используемые признаки: {features}")
        
        # Подготавливаем данные для обучения
        train_data = self.df[self.df['revenue_total'] > 0].copy()
        
        if len(train_data) < 30:
            print(f"Недостаточно данных для обучения ({len(train_data)} записей)")
            return
        
        print(f"Данных для обучения: {len(train_data)} записей")
        
        # Подготавливаем X и y
        X = train_data[features].fillna(0)
        y = train_data['revenue_total']
        
        # Разделение на обучающую и тестовую выборки (временные ряды)
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        print(f"Обучающая выборка: {len(X_train)} записей")
        print(f"Тестовая выборка: {len(X_test)} записей")
        
        # Обучение модели
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Предсказание
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Метрики
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        print(f"Train MAE: {train_mae:,.0f}, Test MAE: {test_mae:,.0f}")
        
        # Показываем коэффициенты
        print(f"\nКоэффициенты модели:")
        for feature, coef in zip(features, self.model.coef_):
            print(f"  {feature}: {coef:.2f}")
        print(f"  Intercept: {self.model.intercept_:.2f}")
        
        print(f"✅ Модель обучена (Test R² = {test_r2:.3f})")
        
        return {
            'features': features,
            'test_r2': test_r2,
            'train_size': len(train_data)
        }
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза revenue_total"""
        print(f"\nСоздание прогноза revenue_total на {forecast_periods} периодов...")
        
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train_revenue_model()")
        
        # Определяем последний временной индекс
        last_time_index = self.df['time_index'].max()
        
        # Получаем уникальные комбинации групп
        if all(col in self.df.columns for col in ['region_to', 'subdivision', 'category']):
            unique_combinations = self.df[['region_to', 'subdivision', 'category']].drop_duplicates()
        else:
            unique_combinations = pd.DataFrame({
                'region_to': ['Unknown'], 
                'subdivision': ['Unknown'], 
                'category': ['Unknown']
            })
        
        print(f"Создание прогноза для {len(unique_combinations)} комбинаций групп...")
        
        # Создаем периоды для прогноза
        forecast_periods_data = []
        for _, combo in unique_combinations.iterrows():
            for i in range(1, forecast_periods + 1):
                period_data = {
                    'time_index': last_time_index + i,
                    'month_sin': np.sin(2 * np.pi * ((last_time_index + i) % 12) / 12),
                    'month_cos': np.cos(2 * np.pi * ((last_time_index + i) % 12) / 12),
                    'region_to': combo['region_to'],
                    'subdivision': combo['subdivision'],
                    'category': combo['category']
                }
                
                # Добавляем квартальные признаки
                month = ((last_time_index + i) % 12) + 1
                quarter = ((month - 1) // 3) + 1
                for q in range(1, 5):
                    period_data[f'q{q}'] = 1 if quarter == q else 0
                
                # Праздничные периоды
                period_data['holiday_period'] = 1 if month in [12, 1, 2, 3, 5] else 0
                
                forecast_periods_data.append(period_data)
        
        self.forecast_df = pd.DataFrame(forecast_periods_data)
        
        # Прогнозируем revenue_total
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
        forecast_features = self.forecast_df[features].fillna(0)
        
        # Прогноз
        predictions = self.model.predict(forecast_features)
        
        # Сохраняем прогноз
        self.forecast_df['revenue_total'] = np.maximum(0, predictions)  # Не допускаем отрицательные значения
        
        print(f"Прогноз создан для {len(forecast_periods_data)} записей")
        print(f"Общий прогноз revenue_total: {self.forecast_df['revenue_total'].sum():,.0f}")
    
    def save_forecast(self, output_file='Simple_Revenue_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.forecast_df is None:
            raise ValueError("Прогноз не создан. Сначала выполните create_forecast()")
        
        # Сохраняем прогноз
        self.forecast_df.to_csv(output_file, index=False)
        print(f"Прогноз сохранен в файл: {output_file}")
        
        return self.forecast_df
    
    def plot_forecast_analysis(self, save_plot=True):
        """Визуализация анализа прогноза"""
        if self.forecast_df is None:
            print("Прогноз не создан")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Исторические данные
        historical_data = self.df[self.df['revenue_total'] > 0]
        if len(historical_data) > 0:
            plt.plot(historical_data['time_index'], historical_data['revenue_total'], 
                    'b-', label='Исторические данные', linewidth=2, alpha=0.7)
        
        # Прогноз
        plt.plot(self.forecast_df['time_index'], self.forecast_df['revenue_total'], 
                'r--', label='Прогноз', linewidth=2, marker='o')
        
        plt.title('Прогноз Revenue Total\n(Простая линейная регрессия)')
        plt.xlabel('Временной индекс')
        plt.ylabel('Revenue Total')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig('simple_revenue_forecast_analysis.png', dpi=300, bbox_inches='tight')
            print("График сохранен как 'simple_revenue_forecast_analysis.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("Простая программа прогнозирования revenue_total")
    print("="*60)
    
    # Инициализация
    forecaster = SimpleRevenueForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка данных
    forecaster.load_data()
    
    # Очистка данных
    forecaster.clean_data()
    
    # Обучение модели
    forecaster.train_revenue_model()
    
    # Создание прогноза
    forecaster.create_forecast(forecast_periods=4)
    
    # Сохранение результатов
    forecaster.save_forecast()
    
    # Визуализация
    forecaster.plot_forecast_analysis()
    
    print("\nПрограмма завершена успешно!")

if __name__ == "__main__":
    main()
