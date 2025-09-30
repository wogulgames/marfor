#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическая программа прогнозирования revenue_total с определением размерностей
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AutoRevenueForecaster:
    def __init__(self, csv_file=None):
        """Инициализация автоматического прогнозировщика revenue"""
        self.csv_file = csv_file
        self.df = None
        self.forecast_df = None
        self.model = None
        
    def load_and_analyze_data(self, csv_file=None):
        """Загрузка и анализ данных"""
        if csv_file:
            self.csv_file = csv_file
            
        if not self.csv_file:
            raise ValueError("Не указан файл для загрузки")
            
        print(f"Загрузка данных из {self.csv_file}...")
        
        # Пробуем разные разделители
        separators = [',', ';', '\t', '|']
        for sep in separators:
            try:
                self.df = pd.read_csv(self.csv_file, sep=sep)
                if len(self.df.columns) > 1:  # Если больше одной колонки, значит разделитель правильный
                    print(f"✅ Загружено с разделителем '{sep}': {len(self.df)} записей, {len(self.df.columns)} колонок")
                    break
            except:
                continue
        
        if self.df is None or len(self.df.columns) <= 1:
            print("❌ Не удалось загрузить файл с правильным разделителем")
            return None
        
        # Анализ размерностей
        print(f"\n📊 АНАЛИЗ РАЗМЕРНОСТЕЙ:")
        print(f"  Строк: {len(self.df)}")
        print(f"  Колонок: {len(self.df.columns)}")
        print(f"  Колонки: {list(self.df.columns)}")
        
        # Проверяем наличие нужных колонок
        required_cols = ['year', 'month', 'revenue_total']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"⚠️  Отсутствуют колонки: {missing_cols}")
            # Ищем похожие колонки
            for col in missing_cols:
                similar_cols = [c for c in self.df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                if similar_cols:
                    print(f"    Возможно имелось в виду: {similar_cols}")
        else:
            print(f"✅ Все необходимые колонки найдены")
        
        return self.df
    
    def clean_data(self):
        """Очистка и подготовка данных"""
        print(f"\n🧹 ОЧИСТКА ДАННЫХ:")
        
        # Очистка временных колонок
        for col in ['year', 'month']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"  {col}: очищено")
        
        # Удаляем строки с пустыми временными данными
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['year', 'month'])
        print(f"  Удалено {initial_count - len(self.df)} записей с пустыми временными данными")
        
        # Очистка revenue_total
        if 'revenue_total' in self.df.columns:
            # Заменяем запятые на пустую строку и конвертируем в числа
            self.df['revenue_total'] = self.df['revenue_total'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['revenue_total'] = pd.to_numeric(self.df['revenue_total'], errors='coerce')
            # Заменяем NaN на 0
            self.df['revenue_total'] = self.df['revenue_total'].fillna(0)
            print(f"  revenue_total: очищено")
        
        print(f"  После очистки: {len(self.df)} записей")
        
        # Анализ данных
        if 'revenue_total' in self.df.columns:
            revenue_stats = self.df['revenue_total'].describe()
            print(f"\n📈 СТАТИСТИКА REVENUE_TOTAL:")
            print(f"  Среднее: {revenue_stats['mean']:,.0f}")
            print(f"  Медиана: {revenue_stats['50%']:,.0f}")
            print(f"  Максимум: {revenue_stats['max']:,.0f}")
            print(f"  Ненулевых записей: {(self.df['revenue_total'] > 0).sum()}")
        
        # Анализ временного диапазона
        if 'year' in self.df.columns and 'month' in self.df.columns:
            print(f"\n📅 ВРЕМЕННОЙ ДИАПАЗОН:")
            print(f"  С {self.df['year'].min()}.{self.df['month'].min():02d} по {self.df['year'].max()}.{self.df['month'].max():02d}")
            print(f"  Всего месяцев: {len(self.df.groupby(['year', 'month']))}")
    
    def prepare_time_features(self):
        """Подготовка временных признаков"""
        print(f"\n⏰ ПОДГОТОВКА ВРЕМЕННЫХ ПРИЗНАКОВ:")
        
        # Создаем временной индекс
        self.df['time_index'] = (self.df['year'] - self.df['year'].min()) * 12 + (self.df['month'] - 1)
        print(f"  time_index: создан")
        
        # Сезонные признаки
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            print(f"  month_sin, month_cos: созданы")
            
            # Квартальные dummy переменные
            self.df['quarter'] = ((self.df['month'] - 1) // 3) + 1
            for q in range(1, 5):
                self.df[f'q{q}'] = (self.df['quarter'] == q).astype(int)
            print(f"  q1, q2, q3, q4: созданы")
        
        # Праздничные периоды
        if 'month' in self.df.columns:
            self.df['holiday_period'] = (
                (self.df['month'] == 12) |  # Декабрь
                (self.df['month'] == 1) |   # Январь
                (self.df['month'] == 2) |   # Февраль
                (self.df['month'] == 3) |   # Март
                (self.df['month'] == 5)     # Май
            ).astype(int)
            print(f"  holiday_period: создан")
        
        print(f"  Временные признаки подготовлены")
    
    def train_revenue_model(self, test_size=0.3):
        """Обучение модели для revenue_total"""
        print(f"\n🤖 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ REVENUE_TOTAL:")
        
        # Подготавливаем данные
        self.prepare_time_features()
        
        # Базовые временные признаки
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
        
        print(f"  Используемые признаки: {features}")
        
        # Подготавливаем данные для обучения
        train_data = self.df[self.df['revenue_total'] > 0].copy()
        
        if len(train_data) < 30:
            print(f"❌ Недостаточно данных для обучения ({len(train_data)} записей)")
            return None
        
        print(f"  Данных для обучения: {len(train_data)} записей")
        
        # Подготавливаем X и y
        X = train_data[features].fillna(0)
        y = train_data['revenue_total']
        
        # Разделение на обучающую и тестовую выборки (временные ряды)
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        print(f"  Обучающая выборка: {len(X_train)} записей")
        print(f"  Тестовая выборка: {len(X_test)} записей")
        
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
        
        print(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        print(f"  Train MAE: {train_mae:,.0f}, Test MAE: {test_mae:,.0f}")
        
        # Показываем коэффициенты
        print(f"\n  📊 КОЭФФИЦИЕНТЫ МОДЕЛИ:")
        for feature, coef in zip(features, self.model.coef_):
            print(f"    {feature}: {coef:.2f}")
        print(f"    Intercept: {self.model.intercept_:.2f}")
        
        print(f"  ✅ Модель обучена (Test R² = {test_r2:.3f})")
        
        return {
            'features': features,
            'test_r2': test_r2,
            'train_size': len(train_data)
        }
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза revenue_total"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА REVENUE_TOTAL НА {forecast_periods} ПЕРИОДОВ:")
        
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train_revenue_model()")
        
        # Определяем последний временной индекс
        last_time_index = self.df['time_index'].max()
        last_year = self.df['year'].max()
        last_month = self.df['month'].max()
        
        print(f"  Последний период: {last_year}.{last_month:02d}")
        print(f"  Прогнозируем до: {last_year}.{last_month + forecast_periods:02d}")
        
        # Получаем уникальные комбинации групп
        group_cols = ['region_to', 'subdivision', 'category']
        available_group_cols = [col for col in group_cols if col in self.df.columns]
        
        if available_group_cols:
            unique_combinations = self.df[available_group_cols].drop_duplicates()
            print(f"  Создание прогноза для {len(unique_combinations)} комбинаций групп")
        else:
            unique_combinations = pd.DataFrame({'dummy': [1]})
            print(f"  Создание прогноза без группировки")
        
        # Создаем периоды для прогноза
        forecast_periods_data = []
        for _, combo in unique_combinations.iterrows():
            for i in range(1, forecast_periods + 1):
                period_data = {
                    'time_index': last_time_index + i,
                    'month_sin': np.sin(2 * np.pi * ((last_time_index + i) % 12) / 12),
                    'month_cos': np.cos(2 * np.pi * ((last_time_index + i) % 12) / 12),
                }
                
                # Добавляем групповые данные
                for col in available_group_cols:
                    period_data[col] = combo[col]
                
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
        
        print(f"  Прогноз создан для {len(forecast_periods_data)} записей")
        print(f"  Общий прогноз revenue_total: {self.forecast_df['revenue_total'].sum():,.0f} ₽")
        
        # Анализ прогноза
        print(f"\n  📊 АНАЛИЗ ПРОГНОЗА:")
        print(f"    Средний прогноз на запись: {self.forecast_df['revenue_total'].mean():,.0f} ₽")
        print(f"    Максимальный прогноз: {self.forecast_df['revenue_total'].max():,.0f} ₽")
        print(f"    Минимальный прогноз: {self.forecast_df['revenue_total'].min():,.0f} ₽")
    
    def save_forecast(self, output_file='Auto_Revenue_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.forecast_df is None:
            raise ValueError("Прогноз не создан. Сначала выполните create_forecast()")
        
        # Сохраняем прогноз
        self.forecast_df.to_csv(output_file, index=False)
        print(f"\n💾 Прогноз сохранен в файл: {output_file}")
        
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
        
        plt.title('Прогноз Revenue Total\n(Автоматическая линейная регрессия)')
        plt.xlabel('Временной индекс')
        plt.ylabel('Revenue Total')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig('auto_revenue_forecast_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 График сохранен как 'auto_revenue_forecast_analysis.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("🚀 АВТОМАТИЧЕСКАЯ ПРОГРАММА ПРОГНОЗИРОВАНИЯ REVENUE_TOTAL")
    print("="*60)
    
    # Инициализация
    forecaster = AutoRevenueForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и анализ данных
    forecaster.load_and_analyze_data()
    
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
    
    print(f"\n🎉 Программа завершена успешно!")

if __name__ == "__main__":
    main()
