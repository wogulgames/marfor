#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная программа прогнозирования маркетинговых данных
с использованием множественной регрессии и сезонных трендов
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketingForecaster:
    def __init__(self, csv_file):
        """Инициализация прогнозировщика"""
        self.csv_file = csv_file
        self.df = None
        self.forecast_df = None
        self.models = {}
        self.scalers = {}
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        self.df = pd.read_csv(self.csv_file)
        print(f"Загружено {len(self.df)} записей")
        
        # Очистка данных
        self.df = self.df.dropna(subset=['year', 'month'])
        self.df['year'] = self.df['year'].astype(int)
        self.df['month'] = self.df['month'].astype(int)
        
        # Очистка числовых колонок
        numeric_columns = [
            'revenue_total', 'revenue_first_transactions', 'revenue_repeat_transactions',
            'traffic_total', 'first_traffic', 'repeat_traffic',
            'transacitons_total', 'first_transactions', 'repeat_transactions',
            'ads_cost', 'not_ads_cost', 'bonus_company', 'promo_cost', 'mar_cost',
            'Сумма по полю distributed_ads_cost'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Создаем временные признаки
        self.df['time_index'] = (self.df['year'] - 2023) * 12 + (self.df['month'] - 9)
        
        # Сезонные признаки
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['quarter'] = ((self.df['month'] - 1) // 3) + 1
        
        # Квартальные признаки
        self.df['q1'] = (self.df['quarter'] == 1).astype(int)
        self.df['q2'] = (self.df['quarter'] == 2).astype(int)
        self.df['q3'] = (self.df['quarter'] == 3).astype(int)
        self.df['q4'] = (self.df['quarter'] == 4).astype(int)
        
        # Праздничные периоды (примерные)
        self.df['holiday_period'] = (
            ((self.df['month'] == 12) & (self.df['month'] >= 20)) |  # Новый год
            ((self.df['month'] == 1) & (self.df['month'] <= 10)) |   # Новогодние каникулы
            ((self.df['month'] == 2) & (self.df['month'] >= 14) & (self.df['month'] <= 23)) |  # 23 февраля
            ((self.df['month'] == 3) & (self.df['month'] >= 1) & (self.df['month'] <= 10)) |   # 8 марта
            ((self.df['month'] == 5) & (self.df['month'] >= 1) & (self.df['month'] <= 10))     # 9 мая
        ).astype(int)
        
        print(f"Период данных: {self.df['year'].min()}-{self.df['month'].min():02d} - {self.df['year'].max()}-{self.df['month'].max():02d}")
        
    def create_forecast_months(self):
        """Создание месяцев для прогноза"""
        print("Подготовка месяцев для прогноза...")
        
        # Определяем последний месяц в данных
        last_year = self.df['year'].max()
        last_month = self.df[self.df['year'] == last_year]['month'].max()
        
        # Создаем месяцы для прогноза (сентябрь-декабрь 2026)
        forecast_months = []
        for month in [9, 10, 11, 12]:
            time_index = (2026 - 2023) * 12 + (month - 9)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            quarter = ((month - 1) // 3) + 1
            
            forecast_months.append({
                'year': 2026,
                'month': month,
                'time_index': time_index,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'quarter': quarter,
                'q1': 1 if quarter == 1 else 0,
                'q2': 1 if quarter == 2 else 0,
                'q3': 1 if quarter == 3 else 0,
                'q4': 1 if quarter == 4 else 0,
                'holiday_period': 1 if month == 12 else 0
            })
        
        self.forecast_months = pd.DataFrame(forecast_months)
        print(f"Будет прогнозироваться {len(forecast_months)} месяцев")
        
    def create_interaction_features(self, df):
        """Создание признаков взаимодействия между метриками"""
        # Логарифмические признаки для нелинейных зависимостей
        for col in ['revenue_total', 'traffic_total', 'ads_cost', 'mar_cost']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])  # log(1+x) для избежания log(0)
        
        # Взаимодействие между временем и сезонностью
        df['time_seasonal'] = df['time_index'] * df['month_sin']
        
        # Взаимодействие между трафиком и временем
        if 'traffic_total' in df.columns:
            df['traffic_time'] = df['traffic_total'] * df['time_index']
        
        return df
    
    def train_multiple_regression_models(self):
        """Обучение моделей множественной регрессии для каждой метрики"""
        print("Обучение моделей множественной регрессии...")
        
        # Метрики для прогнозирования
        target_metrics = [
            'revenue_total', 'revenue_first_transactions', 'revenue_repeat_transactions',
            'traffic_total', 'first_traffic', 'repeat_traffic',
            'transacitons_total', 'first_transactions', 'repeat_transactions',
            'ads_cost', 'not_ads_cost', 'bonus_company', 'promo_cost', 'mar_cost',
            'Сумма по полю distributed_ads_cost'
        ]
        
        # Базовые признаки
        base_features = [
            'time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period'
        ]
        
        for metric in target_metrics:
            if metric not in self.df.columns:
                continue
                
            print(f"Обучение модели для {metric}...")
            
            # Создаем копию данных с дополнительными признаками
            df_model = self.df.copy()
            df_model = self.create_interaction_features(df_model)
            
            # Подготавливаем данные для обучения
            # Используем только записи с ненулевыми значениями целевой метрики
            train_data = df_model[df_model[metric] > 0].copy()
            
            if len(train_data) < 10:  # Минимум 10 записей для обучения
                print(f"  Недостаточно данных для {metric}, пропускаем")
                continue
            
            # Создаем признаки для модели
            feature_columns = base_features.copy()
            
            # Добавляем связанные метрики как признаки
            if metric == 'revenue_total':
                related_metrics = ['traffic_total', 'transacitons_total', 'ads_cost']
            elif metric == 'traffic_total':
                related_metrics = ['ads_cost', 'mar_cost']
            elif metric == 'ads_cost':
                related_metrics = ['traffic_total', 'revenue_total', 'mar_cost']
            elif metric == 'mar_cost':
                related_metrics = ['ads_cost', 'revenue_total']
            else:
                related_metrics = ['revenue_total', 'traffic_total', 'ads_cost']
            
            # Добавляем связанные метрики, если они есть
            for related in related_metrics:
                if related in df_model.columns:
                    feature_columns.append(related)
                    feature_columns.append(f'{related}_log')
            
            # Убираем дубликаты и проверяем наличие колонок
            feature_columns = list(set(feature_columns))
            available_features = [col for col in feature_columns if col in train_data.columns]
            
            if len(available_features) < 3:
                print(f"  Недостаточно признаков для {metric}, используем базовые")
                available_features = [col for col in base_features if col in train_data.columns]
            
            # Подготавливаем X и y
            X = train_data[available_features].fillna(0)
            y = train_data[metric]
            
            # Нормализация признаков
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Обучение модели
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Оценка качества модели
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            print(f"  R² = {r2:.3f}, записей для обучения: {len(train_data)}")
            
            # Сохраняем модель и скейлер
            self.models[metric] = {
                'model': model,
                'features': available_features,
                'r2': r2,
                'train_size': len(train_data)
            }
            self.scalers[metric] = scaler
    
    def create_forecast(self):
        """Создание прогноза с использованием обученных моделей"""
        print("Создание прогноза...")
        
        forecasts = []
        
        # Получаем уникальные комбинации подразделений и категорий
        unique_combinations = self.df[['subdivision', 'category', 'is_paid']].drop_duplicates()
        
        for _, combo in unique_combinations.iterrows():
            subdivision = combo['subdivision']
            category = combo['category']
            is_paid = combo['is_paid']
            
            # Получаем исторические данные для этой комбинации
            historical_data = self.df[
                (self.df['subdivision'] == subdivision) & 
                (self.df['category'] == category) & 
                (self.df['is_paid'] == is_paid)
            ].copy()
            
            if len(historical_data) < 3:  # Минимум 3 записи
                continue
            
            # Создаем прогноз для каждого месяца
            for _, forecast_month in self.forecast_months.iterrows():
                forecast_record = {
                    'year': forecast_month['year'],
                    'month': forecast_month['month'],
                    'Quarter': self._get_quarter(forecast_month['month']),
                    'Halfyear': self._get_halfyear(forecast_month['month']),
                    'region_to': historical_data['region_to'].iloc[0] if len(historical_data) > 0 else 'Unknown',
                    'is_global': historical_data['is_global'].iloc[0] if len(historical_data) > 0 else False,
                    'subdivision': subdivision,
                    'is_paid': is_paid,
                    'category': category,
                    'KEY_clean': f"{subdivision}_{category}_{is_paid}",
                    'is_flowers': historical_data['is_flowers'].iloc[0] if len(historical_data) > 0 else False
                }
                
                # Прогнозируем каждую метрику
                for metric, model_info in self.models.items():
                    try:
                        # Создаем признаки для прогноза
                        forecast_features = forecast_month.copy()
                        
                        # Добавляем средние значения связанных метрик из исторических данных
                        if metric == 'revenue_total':
                            related_metrics = ['traffic_total', 'transacitons_total', 'ads_cost']
                        elif metric == 'traffic_total':
                            related_metrics = ['ads_cost', 'mar_cost']
                        elif metric == 'ads_cost':
                            related_metrics = ['traffic_total', 'revenue_total', 'mar_cost']
                        elif metric == 'mar_cost':
                            related_metrics = ['ads_cost', 'revenue_total']
                        else:
                            related_metrics = ['revenue_total', 'traffic_total', 'ads_cost']
                        
                        # Заполняем связанные метрики средними значениями из исторических данных
                        for related in related_metrics:
                            if related in historical_data.columns:
                                avg_value = historical_data[related].mean()
                                forecast_features[related] = avg_value
                                forecast_features[f'{related}_log'] = np.log1p(avg_value)
                        
                        # Создаем дополнительные признаки
                        forecast_features = self.create_interaction_features(
                            pd.DataFrame([forecast_features])
                        ).iloc[0]
                        
                        # Подготавливаем признаки для модели
                        available_features = model_info['features']
                        X_forecast = np.array([forecast_features[col] if col in forecast_features else 0 
                                             for col in available_features]).reshape(1, -1)
                        
                        # Нормализация
                        X_forecast_scaled = self.scalers[metric].transform(X_forecast)
                        
                        # Прогноз
                        prediction = model_info['model'].predict(X_forecast_scaled)[0]
                        forecast_record[metric] = max(0, prediction)  # Не допускаем отрицательные значения
                        
                    except Exception as e:
                        print(f"Ошибка при прогнозировании {metric} для {subdivision}-{category}: {e}")
                        forecast_record[metric] = 0
                
                forecasts.append(forecast_record)
        
        self.forecast_df = pd.DataFrame(forecasts)
        print(f"Создан прогноз для {len(forecasts)} записей")
        
    def _get_quarter(self, month):
        """Определение квартала по месяцу"""
        if month in [1, 2, 3]:
            return 'Q1'
        elif month in [4, 5, 6]:
            return 'Q2'
        elif month in [7, 8, 9]:
            return 'Q3'
        else:
            return 'Q4'
    
    def _get_halfyear(self, month):
        """Определение полугодия по месяцу"""
        if month in [1, 2, 3, 4, 5, 6]:
            return 'H1'
        else:
            return 'H2'
    
    def fix_logical_inconsistencies(self):
        """Исправление логических несоответствий в прогнозе"""
        print("Исправление логических несоответствий...")
        
        # Исправляем случаи, когда есть трафик, но нет транзакций
        traffic_no_trans = (self.forecast_df['traffic_total'] > 0) & (self.forecast_df['transacitons_total'] == 0)
        if traffic_no_trans.any():
            # Рассчитываем среднюю конверсию из исторических данных
            historical_data = self.df[
                (self.df['traffic_total'] > 0) & 
                (self.df['transacitons_total'] > 0)
            ]
            if len(historical_data) > 0:
                avg_conversion = historical_data['transacitons_total'].sum() / historical_data['traffic_total'].sum()
                self.forecast_df.loc[traffic_no_trans, 'transacitons_total'] = (
                    self.forecast_df.loc[traffic_no_trans, 'traffic_total'] * avg_conversion
                )
                print(f"Исправлено {traffic_no_trans.sum()} записей с трафиком без транзакций")
        
        # Исправляем случаи, когда есть транзакции, но нет first_transactions
        trans_no_first = (self.forecast_df['transacitons_total'] > 0) & (self.forecast_df['first_transactions'] == 0)
        if trans_no_first.any():
            # Рассчитываем среднее соотношение из исторических данных
            historical_data = self.df[
                (self.df['transacitons_total'] > 0) & 
                (self.df['first_transactions'] > 0) & 
                (self.df['repeat_transactions'] > 0)
            ]
            if len(historical_data) > 0:
                first_ratio = historical_data['first_transactions'].sum() / historical_data['transacitons_total'].sum()
                repeat_ratio = historical_data['repeat_transactions'].sum() / historical_data['transacitons_total'].sum()
                
                self.forecast_df.loc[trans_no_first, 'first_transactions'] = (
                    self.forecast_df.loc[trans_no_first, 'transacitons_total'] * first_ratio
                )
                self.forecast_df.loc[trans_no_first, 'repeat_transactions'] = (
                    self.forecast_df.loc[trans_no_first, 'transacitons_total'] * repeat_ratio
                )
                print(f"Исправлено {trans_no_first.sum()} записей с транзакциями без first_transactions")
        
        # Исправляем случаи, когда есть выручка, но нет транзакций
        revenue_no_trans = (self.forecast_df['revenue_total'] > 0) & (self.forecast_df['transacitons_total'] == 0)
        if revenue_no_trans.any():
            # Рассчитываем средний чек из исторических данных
            historical_data = self.df[
                (self.df['revenue_total'] > 0) & 
                (self.df['transacitons_total'] > 0)
            ]
            if len(historical_data) > 0:
                avg_check = historical_data['revenue_total'].sum() / historical_data['transacitons_total'].sum()
                self.forecast_df.loc[revenue_no_trans, 'transacitons_total'] = (
                    self.forecast_df.loc[revenue_no_trans, 'revenue_total'] / avg_check
                )
                print(f"Исправлено {revenue_no_trans.sum()} записей с выручкой без транзакций")
    
    def save_forecast(self, output_file='Marketing_Budget_Forecast_Advanced.csv'):
        """Сохранение прогноза"""
        # Объединяем исторические данные с прогнозом
        combined_df = pd.concat([self.df, self.forecast_df], ignore_index=True)
        
        # Сортируем по времени
        combined_df = combined_df.sort_values(['year', 'month', 'subdivision', 'category'])
        
        # Сохраняем
        combined_df.to_csv(output_file, index=False)
        print(f"Прогноз сохранен в файл: {output_file}")
        print(f"Общее количество записей: {len(combined_df)}")
        print(f"Новых прогнозных записей: {len(self.forecast_df)}")
        
        return combined_df
    
    def generate_summary_report(self):
        """Генерация сводного отчета"""
        print("\n" + "="*50)
        print("СВОДНЫЙ ОТЧЕТ ПО ПРОГНОЗУ")
        print("="*50)
        
        forecast_data = self.forecast_df
        
        print(f"Период прогноза: 2026-09 - 2026-12")
        print(f"Количество прогнозных записей: {len(forecast_data)}")
        
        # Топ категорий по выручке
        if 'revenue_total' in forecast_data.columns:
            top_categories = forecast_data.groupby('category')['revenue_total'].sum().sort_values(ascending=False).head(10)
            print(f"\nТоп-10 категорий по прогнозируемой выручке:")
            for category, revenue in top_categories.items():
                print(f"  {category}: {revenue:,.0f}")
        
        # Подразделения по трафику
        if 'traffic_total' in forecast_data.columns:
            subdivision_traffic = forecast_data.groupby('subdivision')['traffic_total'].sum().sort_values(ascending=False)
            print(f"\nПодразделения по прогнозируемому трафику:")
            for subdivision, traffic in subdivision_traffic.items():
                print(f"  {subdivision}: {traffic:,.0f}")
        
        print("="*50)

def main():
    """Основная функция"""
    print("Улучшенная программа прогнозирования маркетинговых данных")
    print("="*50)
    
    # Инициализация
    forecaster = AdvancedMarketingForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и подготовка данных
    forecaster.load_and_prepare_data()
    
    # Создание месяцев для прогноза
    forecaster.create_forecast_months()
    
    # Обучение моделей
    forecaster.train_multiple_regression_models()
    
    # Создание прогноза
    forecaster.create_forecast()
    
    # Исправление логических несоответствий
    forecaster.fix_logical_inconsistencies()
    
    # Сохранение результатов
    combined_df = forecaster.save_forecast()
    
    # Генерация отчета
    forecaster.generate_summary_report()
    
    print("\nПрограмма завершена успешно!")

if __name__ == "__main__":
    main()
