#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидация каскадной модели на периоде октябрь-декабрь 2025 года
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CascadedForecastValidator:
    def __init__(self, csv_file=None):
        """Инициализация валидатора каскадной модели"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.region_models = {}
        self.validation_forecast = None
        self.actual_data = None
        
    def load_and_analyze_data(self, csv_file=None):
        """Загрузка и анализ данных"""
        if csv_file:
            self.csv_file = csv_file
            
        if not self.csv_file:
            raise ValueError("Не указан файл для загрузки")
            
        print(f"📁 Загрузка данных из {self.csv_file}...")
        
        # Пробуем разные разделители
        separators = [',', ';', '\t', '|']
        for sep in separators:
            try:
                self.df = pd.read_csv(self.csv_file, sep=sep)
                if len(self.df.columns) > 1:
                    print(f"✅ Загружено с разделителем '{sep}': {len(self.df)} записей, {len(self.df.columns)} колонок")
                    break
            except:
                continue
        
        if self.df is None or len(self.df.columns) <= 1:
            print("❌ Не удалось загрузить файл с правильным разделителем")
            return None
        
        return self.df
    
    def clean_data(self):
        """Очистка и подготовка данных"""
        print(f"\n🧹 ОЧИСТКА ДАННЫХ:")
        
        # Очистка временных колонок
        for col in ['year', 'month']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Удаляем строки с пустыми временными данными
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['year', 'month'])
        print(f"  Удалено {initial_count - len(self.df)} записей с пустыми временными данными")
        
        # Очистка revenue_total
        if 'revenue_total' in self.df.columns:
            self.df['revenue_total'] = self.df['revenue_total'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['revenue_total'] = pd.to_numeric(self.df['revenue_total'], errors='coerce')
            self.df['revenue_total'] = self.df['revenue_total'].fillna(0)
        
        print(f"  После очистки: {len(self.df)} записей")
    
    def prepare_training_data(self, train_end_year=2025, train_end_month=9):
        """Подготовка обучающих данных (до сентября 2025)"""
        print(f"\n📚 ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ:")
        print(f"  Обучающие данные: до {train_end_year}.{train_end_month:02d}")
        
        # Обучающие данные (до сентября 2025)
        train_mask = (
            (self.df['year'] < train_end_year) | 
            ((self.df['year'] == train_end_year) & (self.df['month'] <= train_end_month))
        )
        train_df = self.df[train_mask].copy()
        
        print(f"  Обучающих записей: {len(train_df)}")
        
        # Агрегируем данные по регионам и месяцам
        if 'region_to' in train_df.columns and 'revenue_total' in train_df.columns:
            self.aggregated_df = train_df.groupby(['year', 'month', 'region_to'])['revenue_total'].agg([
                'sum', 'mean', 'count', 'std'
            ]).reset_index()
            
            # Переименовываем колонки
            self.aggregated_df.columns = ['year', 'month', 'region_to', 'total_revenue', 'avg_revenue', 'count', 'std_revenue']
            
            print(f"  Агрегировано {len(self.aggregated_df)} записей по регионам")
            
            # Анализ обучающих данных
            train_revenue = self.aggregated_df['total_revenue'].sum()
            print(f"  Общая выручка в обучающих данных: {train_revenue:,.0f} ₽")
            
            # Анализ по регионам
            region_totals = self.aggregated_df.groupby('region_to')['total_revenue'].sum().sort_values(ascending=False)
            print(f"  Выручка по регионам в обучающих данных:")
            for region, total in region_totals.items():
                print(f"    {region}: {total:,.0f} ₽")
    
    def build_region_models(self):
        """Построение моделей для каждого региона на обучающих данных"""
        print(f"\n🤖 ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ РЕГИОНОВ:")
        
        if self.aggregated_df is not None:
            for region in self.aggregated_df['region_to'].unique():
                region_data = self.aggregated_df[self.aggregated_df['region_to'] == region].copy()
                
                if len(region_data) > 6:  # Минимум 6 месяцев данных
                    print(f"\n  🔧 Обучение модели для {region}:")
                    
                    # Подготавливаем данные
                    region_data = region_data.sort_values(['year', 'month'])
                    region_data['time_index'] = (region_data['year'] - region_data['year'].min()) * 12 + (region_data['month'] - 1)
                    
                    # Сезонные признаки
                    region_data['month_sin'] = np.sin(2 * np.pi * region_data['month'] / 12)
                    region_data['month_cos'] = np.cos(2 * np.pi * region_data['month'] / 12)
                    
                    # Квартальные признаки
                    region_data['quarter'] = ((region_data['month'] - 1) // 3) + 1
                    for q in range(1, 5):
                        region_data[f'q{q}'] = (region_data['quarter'] == q).astype(int)
                    
                    # Праздничные периоды
                    region_data['holiday_period'] = (
                        (region_data['month'] == 12) |  # Декабрь
                        (region_data['month'] == 1) |   # Январь
                        (region_data['month'] == 2) |   # Февраль
                        (region_data['month'] == 3) |   # Март
                        (region_data['month'] == 5)     # Май
                    ).astype(int)
                    
                    # Подготавливаем X и y
                    features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
                    X = region_data[features].fillna(0)
                    y = region_data['total_revenue']
                    
                    # Обучение модели
                    model = Ridge(alpha=1.0)
                    model.fit(X, y)
                    
                    # Предсказание
                    y_pred = model.predict(X)
                    
                    # Метрики
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    print(f"    R²: {r2:.3f}")
                    print(f"    MAE: {mae:,.0f}")
                    
                    # Сохраняем модель
                    self.region_models[region] = {
                        'model': model,
                        'features': features,
                        'r2': r2,
                        'mae': mae,
                        'data': region_data
                    }
                    
                    if r2 > 0.3:
                        print(f"    ✅ Модель хорошего качества")
                    elif r2 > 0.1:
                        print(f"    ⚠️  Модель удовлетворительного качества")
                    else:
                        print(f"    ❌ Модель слабого качества")
    
    def create_validation_forecast(self, forecast_year=2025, forecast_months=[10, 11, 12]):
        """Создание прогноза на период октябрь-декабрь 2025"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА НА ПЕРИОД {forecast_year}.{forecast_months[0]:02d}-{forecast_year}.{forecast_months[-1]:02d}:")
        
        if not self.region_models:
            print("❌ Нет обученных моделей для прогнозирования")
            return None
        
        # Определяем последний временной индекс из обучающих данных
        last_year = self.aggregated_df['year'].max()
        last_month = self.aggregated_df['month'].max()
        
        print(f"  Последний период в обучающих данных: {last_year}.{last_month:02d}")
        
        # Создаем прогноз для каждого региона
        forecast_data = []
        
        for region, model_info in self.region_models.items():
            print(f"\n  📊 Прогноз для {region}:")
            
            # Получаем последние данные региона
            region_data = model_info['data']
            last_time_index = region_data['time_index'].max()
            
            # Создаем периоды для прогноза
            for i, month in enumerate(forecast_months):
                period_data = {
                    'year': forecast_year,
                    'month': month,
                    'region_to': region,
                    'time_index': last_time_index + i + 1,
                    'month_sin': np.sin(2 * np.pi * month / 12),
                    'month_cos': np.cos(2 * np.pi * month / 12),
                }
                
                # Добавляем квартальные признаки
                quarter = ((month - 1) // 3) + 1
                for q in range(1, 5):
                    period_data[f'q{q}'] = 1 if quarter == q else 0
                
                # Праздничные периоды
                period_data['holiday_period'] = 1 if month in [12, 1, 2, 3, 5] else 0
                
                # Прогноз
                features = model_info['features']
                X_forecast = np.array([period_data[feature] for feature in features]).reshape(1, -1)
                forecast_value = model_info['model'].predict(X_forecast)[0]
                
                period_data['forecast_revenue'] = max(0, forecast_value)  # Не допускаем отрицательные значения
                
                forecast_data.append(period_data)
                
                print(f"    {forecast_year}.{month:02d}: {forecast_value:,.0f} ₽")
        
        # Создаем DataFrame с прогнозом
        self.validation_forecast = pd.DataFrame(forecast_data)
        
        # Анализ прогноза
        total_forecast = self.validation_forecast['forecast_revenue'].sum()
        print(f"\n  📊 ОБЩИЙ ПРОГНОЗ: {total_forecast:,.0f} ₽")
        
        # Анализ по регионам
        print(f"\n  🌍 ПРОГНОЗ ПО РЕГИОНАМ:")
        region_forecasts = self.validation_forecast.groupby('region_to')['forecast_revenue'].sum().sort_values(ascending=False)
        for region, forecast in region_forecasts.items():
            print(f"    {region}: {forecast:,.0f} ₽")
        
        return self.validation_forecast
    
    def get_actual_data(self, actual_year=2025, actual_months=[10, 11, 12]):
        """Получение фактических данных за период октябрь-декабрь 2025"""
        print(f"\n📊 ПОЛУЧЕНИЕ ФАКТИЧЕСКИХ ДАННЫХ ЗА ПЕРИОД {actual_year}.{actual_months[0]:02d}-{actual_year}.{actual_months[-1]:02d}:")
        
        # Фильтруем фактические данные
        actual_mask = (
            (self.df['year'] == actual_year) & 
            (self.df['month'].isin(actual_months))
        )
        actual_df = self.df[actual_mask].copy()
        
        if len(actual_df) == 0:
            print("❌ Нет фактических данных за указанный период")
            return None
        
        # Агрегируем фактические данные по регионам
        if 'region_to' in actual_df.columns and 'revenue_total' in actual_df.columns:
            self.actual_data = actual_df.groupby(['year', 'month', 'region_to'])['revenue_total'].sum().reset_index()
            self.actual_data.columns = ['year', 'month', 'region_to', 'actual_revenue']
            
            print(f"  Фактических записей: {len(self.actual_data)}")
            
            # Анализ фактических данных
            total_actual = self.actual_data['actual_revenue'].sum()
            print(f"  📊 ОБЩАЯ ФАКТИЧЕСКАЯ ВЫРУЧКА: {total_actual:,.0f} ₽")
            
            # Анализ по регионам
            print(f"\n  🌍 ФАКТИЧЕСКАЯ ВЫРУЧКА ПО РЕГИОНАМ:")
            region_actuals = self.actual_data.groupby('region_to')['actual_revenue'].sum().sort_values(ascending=False)
            for region, actual in region_actuals.items():
                print(f"    {region}: {actual:,.0f} ₽")
            
            return self.actual_data
        else:
            print("❌ Не найдены необходимые колонки для анализа фактических данных")
            return None
    
    def compare_forecast_vs_actual(self):
        """Сравнение прогноза с фактическими данными"""
        print(f"\n📈 СРАВНЕНИЕ ПРОГНОЗА С ФАКТИЧЕСКИМИ ДАННЫМИ:")
        
        if self.validation_forecast is None or self.actual_data is None:
            print("❌ Нет данных для сравнения")
            return None
        
        # Объединяем прогноз и фактические данные
        comparison = pd.merge(
            self.validation_forecast[['region_to', 'month', 'forecast_revenue']],
            self.actual_data[['region_to', 'month', 'actual_revenue']],
            on=['region_to', 'month'],
            how='inner'
        )
        
        if len(comparison) == 0:
            print("❌ Нет совпадающих данных для сравнения")
            return None
        
        # Агрегируем по регионам
        region_comparison = comparison.groupby('region_to').agg({
            'forecast_revenue': 'sum',
            'actual_revenue': 'sum'
        }).reset_index()
        
        # Вычисляем метрики
        region_comparison['error'] = region_comparison['forecast_revenue'] - region_comparison['actual_revenue']
        region_comparison['error_pct'] = (region_comparison['error'] / region_comparison['actual_revenue'] * 100).round(2)
        region_comparison['mape'] = abs(region_comparison['error_pct'])
        
        print(f"  📊 СРАВНЕНИЕ ПО РЕГИОНАМ:")
        for _, row in region_comparison.iterrows():
            print(f"    {row['region_to']}:")
            print(f"      Прогноз: {row['forecast_revenue']:,.0f} ₽")
            print(f"      Факт: {row['actual_revenue']:,.0f} ₽")
            print(f"      Ошибка: {row['error']:,.0f} ₽ ({row['error_pct']:+.1f}%)")
            print(f"      MAPE: {row['mape']:.1f}%")
        
        # Общие метрики
        total_forecast = region_comparison['forecast_revenue'].sum()
        total_actual = region_comparison['actual_revenue'].sum()
        total_error = total_forecast - total_actual
        total_error_pct = (total_error / total_actual * 100) if total_actual > 0 else 0
        mean_mape = region_comparison['mape'].mean()
        
        print(f"\n  📊 ОБЩИЕ МЕТРИКИ:")
        print(f"    Общий прогноз: {total_forecast:,.0f} ₽")
        print(f"    Общий факт: {total_actual:,.0f} ₽")
        print(f"    Общая ошибка: {total_error:,.0f} ₽ ({total_error_pct:+.1f}%)")
        print(f"    Средний MAPE: {mean_mape:.1f}%")
        
        # Оценка качества
        if mean_mape < 10:
            quality = "ОТЛИЧНОЕ"
        elif mean_mape < 20:
            quality = "ХОРОШЕЕ"
        elif mean_mape < 30:
            quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
        else:
            quality = "СЛАБОЕ"
        
        print(f"    Качество прогноза: {quality}")
        
        return region_comparison
    
    def save_results(self, output_file='Validation_Results.csv'):
        """Сохранение результатов валидации"""
        if self.validation_forecast is None:
            print("❌ Нет данных для сохранения")
            return None
        
        # Сохраняем прогноз
        self.validation_forecast.to_csv(output_file, index=False)
        print(f"\n💾 Результаты валидации сохранены в файл: {output_file}")
        
        return self.validation_forecast

def main():
    """Основная функция для валидации"""
    print("🎯 ВАЛИДАЦИЯ КАСКАДНОЙ МОДЕЛИ")
    print("="*60)
    print("Прогноз на период октябрь-декабрь 2025 года")
    print("="*60)
    
    # Инициализация
    validator = CascadedForecastValidator('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и очистка данных
    validator.load_and_analyze_data()
    validator.clean_data()
    
    # Подготовка обучающих данных (до сентября 2025)
    validator.prepare_training_data()
    
    # Построение моделей
    validator.build_region_models()
    
    # Создание прогноза на октябрь-декабрь 2025
    validator.create_validation_forecast()
    
    # Получение фактических данных
    validator.get_actual_data()
    
    # Сравнение прогноза с фактом
    validator.compare_forecast_vs_actual()
    
    # Сохранение результатов
    validator.save_results()
    
    print(f"\n🎉 Валидация завершена!")

if __name__ == "__main__":
    main()
