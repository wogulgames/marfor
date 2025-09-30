#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенная программа прогнозирования с сезонными трендами и учетом зависимостей между полями
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimplifiedSeasonalForecaster:
    def __init__(self, csv_file=None):
        """Инициализация упрощенного прогнозировщика"""
        self.csv_file = csv_file
        self.df = None
        self.forecast_df = None
        self.models = {}
        self.scalers = {}
        self.dependencies = {}
        
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
        
        # Очистка всех числовых колонок - заменяем запятые и конвертируем в числа
        numeric_cols = ['Сумма по полю distributed_ads_cost', 'revenue_first_transactions', 
                       'revenue_repeat_transactions', 'revenue_total', 'Сумма по полю first_traffic', 
                       'Сумма по полю repeat_traffic', 'traffic_total', 'first_transactions', 
                       'repeat_transactions', 'transacitons_total', 'ads_cost', 'not_ads_cost', 
                       'bonus_company', 'promo_cost', 'mar_cost']
        
        for col in numeric_cols:
            if col in self.df.columns:
                # Заменяем запятые на пустую строку и конвертируем в числа
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Заменяем NaN на 0
                self.df[col] = self.df[col].fillna(0)
        
        print(f"После очистки: {len(self.df)} записей")
        
    def analyze_dependencies(self):
        """Анализ зависимостей между полями"""
        print("\n" + "="*60)
        print("АНАЛИЗ ЗАВИСИМОСТЕЙ МЕЖДУ ПОЛЯМИ")
        print("="*60)
        
        # Определяем зависимости для платных каналов
        paid_data = self.df[self.df['is_paid'] == 'paid'].copy()
        
        if len(paid_data) > 0:
            print(f"\nАнализ платных каналов ({len(paid_data)} записей):")
            
            # ads_cost → traffic
            if 'ads_cost' in paid_data.columns and 'traffic_total' in paid_data.columns:
                ads_traffic_corr = paid_data[paid_data['ads_cost'] > 0]['ads_cost'].corr(
                    paid_data[paid_data['ads_cost'] > 0]['traffic_total'])
                print(f"  Корреляция ads_cost → traffic_total: {ads_traffic_corr:.3f}")
            
            # traffic → transactions
            if 'traffic_total' in paid_data.columns and 'transacitons_total' in paid_data.columns:
                traffic_trans_corr = paid_data[paid_data['traffic_total'] > 0]['traffic_total'].corr(
                    paid_data[paid_data['traffic_total'] > 0]['transacitons_total'])
                print(f"  Корреляция traffic_total → transacitons_total: {traffic_trans_corr:.3f}")
            
            # transactions → revenue
            if 'transacitons_total' in paid_data.columns and 'revenue_total' in paid_data.columns:
                trans_revenue_corr = paid_data[paid_data['transacitons_total'] > 0]['transacitons_total'].corr(
                    paid_data[paid_data['transacitons_total'] > 0]['revenue_total'])
                print(f"  Корреляция transacitons_total → revenue_total: {trans_revenue_corr:.3f}")
        
        # Анализ органических каналов
        organic_data = self.df[self.df['is_paid'] == 'organic'].copy()
        
        if len(organic_data) > 0:
            print(f"\nАнализ органических каналов ({len(organic_data)} записей):")
            
            # Для органических каналов ads_cost должен быть 0 или минимальным
            if 'ads_cost' in organic_data.columns:
                organic_ads_cost = organic_data['ads_cost'].sum()
                print(f"  Общий ads_cost в органических каналах: {organic_ads_cost:,.0f}")
                
                # Проверяем на выбросы
                organic_with_ads = organic_data[organic_data['ads_cost'] > 0]
                if len(organic_with_ads) > 0:
                    print(f"  ⚠️  Найдено {len(organic_with_ads)} записей с ads_cost > 0 в органических каналах")
                    print(f"      Максимальное значение: {organic_with_ads['ads_cost'].max():,.0f}")
    
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
    
    def create_group_features(self, df):
        """Создание признаков для групп"""
        df = df.copy()
        
        # Кодирование категориальных признаков
        if 'region_to' in df.columns:
            df['region_encoded'] = df['region_to'].astype('category').cat.codes
        
        if 'subdivision' in df.columns:
            df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes
        
        if 'category' in df.columns:
            df['category_encoded'] = df['category'].astype('category').cat.codes
        
        if 'is_paid' in df.columns:
            df['is_paid_encoded'] = (df['is_paid'] == 'paid').astype(int)
        
        return df
    
    def train_seasonal_models(self, test_size=0.3):
        """Обучение сезонных моделей для каждого поля"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ СЕЗОННЫХ МОДЕЛЕЙ")
        print("="*60)
        
        # Подготавливаем данные
        self.prepare_time_features()
        self.df = self.create_group_features(self.df)
        
        # Определяем поля для прогнозирования
        target_fields = ['ads_cost', 'traffic_total', 'transacitons_total', 'revenue_total', 
                        'first_transactions', 'repeat_transactions', 'revenue_first_transactions', 
                        'revenue_repeat_transactions', 'bonus_company', 'promo_cost', 'mar_cost']
        
        # Базовые временные признаки
        base_features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
        
        # Добавляем групповые признаки
        group_features = ['region_encoded', 'subdivision_encoded', 'category_encoded', 'is_paid_encoded']
        for feature in group_features:
            if feature in self.df.columns:
                base_features.append(feature)
        
        print(f"Используемые признаки: {base_features}")
        
        for target in target_fields:
            if target not in self.df.columns:
                continue
                
            print(f"\n{'='*50}")
            print(f"Обучение модели для {target}")
            print(f"{'='*50}")
            
            # Подготавливаем данные для обучения
            train_data = self.df[self.df[target] > 0].copy()
            
            if len(train_data) < 30:  # Минимум 30 записей
                print(f"  Недостаточно данных для {target} ({len(train_data)} записей), пропускаем")
                continue
            
            print(f"  Данных для обучения: {len(train_data)} записей")
            
            # Подготавливаем X и y
            X = train_data[base_features].fillna(0)
            y = train_data[target]
            
            # Разделение на обучающую и тестовую выборки (временные ряды)
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            print(f"  Обучающая выборка: {len(X_train)} записей")
            print(f"  Тестовая выборка: {len(X_test)} записей")
            
            # Нормализация
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Обучение модели (используем Ridge для стабильности)
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            # Предсказание
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Метрики
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
            print(f"  Train MAE: {train_mae:,.0f}, Test MAE: {test_mae:,.0f}")
            
            # Сохраняем модель
            self.models[target] = {
                'model': model,
                'features': base_features,
                'test_r2': test_r2,
                'train_size': len(train_data)
            }
            self.scalers[target] = scaler
            
            print(f"  ✅ Модель обучена (Test R² = {test_r2:.3f})")
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза с учетом зависимостей"""
        print(f"\nСоздание прогноза на {forecast_periods} периодов...")
        
        if not self.models:
            raise ValueError("Модели не обучены. Сначала выполните train_seasonal_models()")
        
        # Определяем последний временной индекс
        last_time_index = self.df['time_index'].max()
        
        # Получаем уникальные комбинации групп
        if all(col in self.df.columns for col in ['region_to', 'subdivision', 'category', 'is_paid']):
            unique_combinations = self.df[['region_to', 'subdivision', 'category', 'is_paid']].drop_duplicates()
        else:
            unique_combinations = pd.DataFrame({
                'region_to': ['Unknown'], 
                'subdivision': ['Unknown'], 
                'category': ['Unknown'],
                'is_paid': ['organic']
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
                    'category': combo['category'],
                    'is_paid': combo['is_paid']
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
        
        # Создаем признаки для прогноза
        self.forecast_df = self.create_group_features(self.forecast_df)
        
        # Прогнозируем каждую метрику
        for target, model_info in self.models.items():
            try:
                # Подготавливаем признаки для прогноза
                forecast_features = self.forecast_df[model_info['features']].fillna(0)
                
                # Нормализация
                forecast_scaled = self.scalers[target].transform(forecast_features)
                
                # Прогноз
                predictions = model_info['model'].predict(forecast_scaled)
                
                # Сохраняем прогноз
                self.forecast_df[target] = np.maximum(0, predictions)  # Не допускаем отрицательные значения
                
                print(f"  {target}: прогноз создан")
                
            except Exception as e:
                print(f"  Ошибка при прогнозировании {target}: {e}")
                self.forecast_df[target] = 0
        
        # Применяем логику зависимостей
        self._apply_dependencies()
        
        print(f"Прогноз создан для {len(forecast_periods_data)} записей")
    
    def _apply_dependencies(self):
        """Применение логики зависимостей между полями"""
        print("Применение логики зависимостей...")
        
        # Для органических каналов: ads_cost должен быть 0
        organic_mask = self.forecast_df['is_paid'] == 'organic'
        self.forecast_df.loc[organic_mask, 'ads_cost'] = 0
        
        # Для платных каналов: применяем зависимости
        paid_mask = self.forecast_df['is_paid'] == 'paid'
        
        if paid_mask.any():
            # Если ads_cost = 0, то traffic должен быть минимальным
            zero_ads_mask = (self.forecast_df['ads_cost'] == 0) & paid_mask
            if zero_ads_mask.any():
                # Устанавливаем минимальный трафик для платных каналов без рекламы
                self.forecast_df.loc[zero_ads_mask, 'traffic_total'] = np.maximum(
                    self.forecast_df.loc[zero_ads_mask, 'traffic_total'], 1)
            
            # Если traffic = 0, то transactions должны быть 0
            zero_traffic_mask = (self.forecast_df['traffic_total'] == 0) & paid_mask
            if zero_traffic_mask.any():
                self.forecast_df.loc[zero_traffic_mask, 'transacitons_total'] = 0
                self.forecast_df.loc[zero_traffic_mask, 'first_transactions'] = 0
                self.forecast_df.loc[zero_traffic_mask, 'repeat_transactions'] = 0
            
            # Если transactions = 0, то revenue должны быть 0
            zero_trans_mask = (self.forecast_df['transacitons_total'] == 0) & paid_mask
            if zero_trans_mask.any():
                self.forecast_df.loc[zero_trans_mask, 'revenue_total'] = 0
                self.forecast_df.loc[zero_trans_mask, 'revenue_first_transactions'] = 0
                self.forecast_df.loc[zero_trans_mask, 'revenue_repeat_transactions'] = 0
        
        print("Логика зависимостей применена")
    
    def save_forecast(self, output_file='Simplified_Seasonal_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.forecast_df is None:
            raise ValueError("Прогноз не создан. Сначала выполните create_forecast()")
        
        # Сохраняем прогноз
        self.forecast_df.to_csv(output_file, index=False)
        print(f"Прогноз сохранен в файл: {output_file}")
        
        return self.forecast_df
    
    def generate_summary_report(self):
        """Генерация сводного отчета"""
        print("\n" + "="*80)
        print("СВОДНЫЙ ОТЧЕТ О КАЧЕСТВЕ МОДЕЛЕЙ")
        print("="*80)
        
        for target, model_info in self.models.items():
            print(f"{target}:")
            print(f"  Модель: Ridge")
            print(f"  Тестовый R²: {model_info['test_r2']:.3f}")
            print(f"  Размер обучающей выборки: {model_info['train_size']}")
        
        print("="*80)
    
    def plot_forecast_analysis(self, target_columns=None, save_plot=True):
        """Визуализация анализа прогноза"""
        if target_columns is None:
            target_columns = list(self.models.keys())[:4]  # Показываем первые 4 метрики
        
        n_plots = len(target_columns)
        if n_plots == 0:
            print("Нет метрик для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, target in enumerate(target_columns):
            if i >= 4:  # Максимум 4 графика
                break
                
            if target not in self.models:
                continue
            
            ax = axes[i]
            
            # Исторические данные
            historical_data = self.df[self.df[target] > 0]
            if len(historical_data) > 0:
                ax.plot(historical_data['time_index'], historical_data[target], 
                       'b-', label='Исторические данные', linewidth=2)
            
            # Прогноз
            if target in self.forecast_df.columns:
                ax.plot(self.forecast_df['time_index'], self.forecast_df[target], 
                       'r--', label='Прогноз', linewidth=2, marker='o')
            
            ax.set_title(f'{target}\nR² = {self.models[target]["test_r2"]:.3f}')
            ax.set_xlabel('Временной индекс')
            ax.set_ylabel('Значение')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Скрываем лишние подграфики
        for i in range(n_plots, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('simplified_seasonal_forecast_analysis.png', dpi=300, bbox_inches='tight')
            print("График сохранен как 'simplified_seasonal_forecast_analysis.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("Упрощенная программа прогнозирования с сезонными трендами")
    print("="*60)
    
    # Инициализация
    forecaster = SimplifiedSeasonalForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка данных
    forecaster.load_data()
    
    # Очистка данных
    forecaster.clean_data()
    
    # Анализ зависимостей
    forecaster.analyze_dependencies()
    
    # Обучение сезонных моделей
    forecaster.train_seasonal_models()
    
    # Создание прогноза
    forecaster.create_forecast(forecast_periods=4)
    
    # Сохранение результатов
    forecaster.save_forecast()
    
    # Генерация отчета
    forecaster.generate_summary_report()
    
    # Визуализация
    forecaster.plot_forecast_analysis()
    
    print("\nПрограмма завершена успешно!")

if __name__ == "__main__":
    main()
