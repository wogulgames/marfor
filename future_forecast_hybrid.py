#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогноз на будущие периоды с использованием гибридной каскадной модели
Период: сентябрь 2025 - декабрь 2026
Сравнение с прогнозными данными аналитиков
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn модели
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet не установлен")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FutureForecastHybrid:
    def __init__(self, csv_file=None):
        """Инициализация прогноза на будущие периоды"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.historical_data = None
        self.analyst_forecast = None
        self.region_stability = {}
        self.selected_models = {}
        self.models = {}
        self.our_forecast = {}
        self.comparison_results = {}
        
    def load_and_clean_data(self, csv_file=None):
        """Загрузка и очистка данных"""
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
        
        # Очистка данных
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
        return self.df
    
    def prepare_aggregated_data(self):
        """Подготовка агрегированных данных"""
        print(f"\n📚 ПОДГОТОВКА АГРЕГИРОВАННЫХ ДАННЫХ:")
        
        # Агрегируем данные по регионам и месяцам
        if 'region_to' in self.df.columns and 'revenue_total' in self.df.columns:
            self.aggregated_df = self.df.groupby(['year', 'month', 'region_to'])['revenue_total'].agg([
                'sum', 'mean', 'count', 'std'
            ]).reset_index()
            
            # Переименовываем колонки
            self.aggregated_df.columns = ['year', 'month', 'region_to', 'total_revenue', 'avg_revenue', 'count', 'std_revenue']
            
            print(f"  Агрегировано {len(self.aggregated_df)} записей по регионам")
            
            # Анализ данных
            total_revenue = self.aggregated_df['total_revenue'].sum()
            print(f"  Общая выручка: {total_revenue:,.0f} ₽")
            
            return self.aggregated_df
        else:
            print("❌ Не найдены необходимые колонки")
            return None
    
    def split_historical_and_forecast_data(self):
        """Разделение данных на исторические и прогнозные"""
        print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"  Исторические данные: до августа 2025")
        print(f"  Прогнозные данные аналитиков: сентябрь 2025 - август 2026")
        print(f"  Наш прогноз: сентябрь 2025 - декабрь 2026")
        
        # Исторические данные (до августа 2025)
        historical_mask = (
            (self.aggregated_df['year'] < 2025) | 
            ((self.aggregated_df['year'] == 2025) & (self.aggregated_df['month'] <= 8))
        )
        self.historical_data = self.aggregated_df[historical_mask].copy()
        
        # Прогнозные данные аналитиков (сентябрь 2025 - август 2026)
        analyst_forecast_mask = (
            ((self.aggregated_df['year'] == 2025) & (self.aggregated_df['month'] >= 9)) |
            ((self.aggregated_df['year'] == 2026) & (self.aggregated_df['month'] <= 8))
        )
        self.analyst_forecast = self.aggregated_df[analyst_forecast_mask].copy()
        
        print(f"  Исторических записей: {len(self.historical_data)}")
        print(f"  Прогнозных записей аналитиков: {len(self.analyst_forecast)}")
        
        # Анализ исторических данных
        historical_revenue = self.historical_data['total_revenue'].sum()
        print(f"  Выручка в исторических данных: {historical_revenue:,.0f} ₽")
        
        # Анализ прогнозных данных аналитиков
        if len(self.analyst_forecast) > 0:
            analyst_revenue = self.analyst_forecast['total_revenue'].sum()
            print(f"  Выручка в прогнозных данных аналитиков: {analyst_revenue:,.0f} ₽")
        else:
            print("  ⚠️  Нет прогнозных данных аналитиков")
    
    def analyze_region_stability(self):
        """Анализ стабильности данных по регионам на исторических данных"""
        print(f"\n🔍 АНАЛИЗ СТАБИЛЬНОСТИ ДАННЫХ ПО РЕГИОНАМ:")
        
        if self.historical_data is None:
            print("❌ Нет исторических данных для анализа")
            return None
        
        for region in self.historical_data['region_to'].unique():
            region_data = self.historical_data[self.historical_data['region_to'] == region].copy()
            
            if len(region_data) > 6:  # Минимум 6 месяцев данных
                print(f"\n  🌍 Регион: {region}")
                
                # Сортируем по времени
                region_data = region_data.sort_values(['year', 'month'])
                
                # Вычисляем метрики стабильности
                revenue_values = region_data['total_revenue'].values
                
                # 1. Коэффициент вариации (CV)
                cv = np.std(revenue_values) / np.mean(revenue_values) if np.mean(revenue_values) > 0 else 0
                
                # 2. Тренд (линейная регрессия)
                time_index = np.arange(len(revenue_values))
                trend_slope = np.polyfit(time_index, revenue_values, 1)[0]
                trend_strength = abs(trend_slope) / np.mean(revenue_values) if np.mean(revenue_values) > 0 else 0
                
                # 3. Сезонность (разброс по месяцам)
                monthly_means = region_data.groupby('month')['total_revenue'].mean()
                seasonality = monthly_means.std() / monthly_means.mean() if monthly_means.mean() > 0 else 0
                
                # 4. Количество выбросов (IQR метод)
                q1, q3 = np.percentile(revenue_values, [25, 75])
                iqr = q3 - q1
                outliers = np.sum((revenue_values < q1 - 1.5 * iqr) | (revenue_values > q3 + 1.5 * iqr))
                outlier_ratio = outliers / len(revenue_values)
                
                # 5. Автокорреляция (lag-1)
                if len(revenue_values) > 1:
                    autocorr = np.corrcoef(revenue_values[:-1], revenue_values[1:])[0, 1]
                    autocorr = 0 if np.isnan(autocorr) else autocorr
                else:
                    autocorr = 0
                
                # Общая оценка стабильности
                stability_score = (
                    (1 - min(cv, 1)) * 0.3 +           # Низкая вариативность
                    (1 - min(trend_strength, 1)) * 0.2 + # Слабый тренд
                    (1 - min(seasonality, 1)) * 0.2 +    # Низкая сезонность
                    (1 - outlier_ratio) * 0.2 +          # Мало выбросов
                    max(autocorr, 0) * 0.1               # Положительная автокорреляция
                )
                
                # Определяем тип региона
                if stability_score > 0.7:
                    region_type = "СТАБИЛЬНЫЙ"
                    recommended_model = "Random Forest"
                elif stability_score > 0.4:
                    region_type = "УМЕРЕННО СТАБИЛЬНЫЙ"
                    recommended_model = "Random Forest"
                else:
                    region_type = "НЕСТАБИЛЬНЫЙ"
                    recommended_model = "Prophet"
                
                self.region_stability[region] = {
                    'cv': cv,
                    'trend_strength': trend_strength,
                    'seasonality': seasonality,
                    'outlier_ratio': outlier_ratio,
                    'autocorr': autocorr,
                    'stability_score': stability_score,
                    'region_type': region_type,
                    'recommended_model': recommended_model
                }
                
                print(f"    Коэффициент вариации: {cv:.3f}")
                print(f"    Сила тренда: {trend_strength:.3f}")
                print(f"    Сезонность: {seasonality:.3f}")
                print(f"    Доля выбросов: {outlier_ratio:.3f}")
                print(f"    Автокорреляция: {autocorr:.3f}")
                print(f"    Оценка стабильности: {stability_score:.3f}")
                print(f"    Тип региона: {region_type}")
                print(f"    Рекомендуемая модель: {recommended_model}")
    
    def prepare_features_for_sklearn(self, data):
        """Подготовка признаков для scikit-learn моделей"""
        data = data.copy()
        data = data.sort_values(['year', 'month'])
        
        # Временной индекс
        data['time_index'] = (data['year'] - data['year'].min()) * 12 + (data['month'] - 1)
        
        # Сезонные признаки
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Квартальные признаки
        data['quarter'] = ((data['month'] - 1) // 3) + 1
        for q in range(1, 5):
            data[f'q{q}'] = (data['quarter'] == q).astype(int)
        
        # Праздничные периоды
        data['holiday_period'] = (
            (data['month'] == 12) |  # Декабрь
            (data['month'] == 1) |   # Январь
            (data['month'] == 2) |   # Февраль
            (data['month'] == 3) |   # Март
            (data['month'] == 5)     # Май
        ).astype(int)
        
        # Полиномиальные признаки
        data['time_squared'] = data['time_index'] ** 2
        data['time_cubed'] = data['time_index'] ** 3
        
        return data
    
    def prepare_features_for_prophet(self, data):
        """Подготовка данных для Prophet"""
        prophet_data = data.copy()
        prophet_data = prophet_data.sort_values(['year', 'month'])
        
        # Создаем дату
        prophet_data['ds'] = pd.to_datetime(prophet_data[['year', 'month']].assign(day=1))
        prophet_data['y'] = prophet_data['total_revenue']
        
        return prophet_data[['ds', 'y']]
    
    def train_hybrid_models(self):
        """Обучение гибридных моделей для каждого региона"""
        print(f"\n🤖 ОБУЧЕНИЕ ГИБРИДНЫХ МОДЕЛЕЙ:")
        
        if self.historical_data is None or not self.region_stability:
            print("❌ Нет исторических данных или анализа стабильности")
            return None
        
        for region in self.historical_data['region_to'].unique():
            region_data = self.historical_data[self.historical_data['region_to'] == region].copy()
            
            if len(region_data) > 6 and region in self.region_stability:
                print(f"\n  🌍 Регион: {region}")
                print(f"    Тип: {self.region_stability[region]['region_type']}")
                print(f"    Рекомендуемая модель: {self.region_stability[region]['recommended_model']}")
                
                # Выбираем модель на основе стабильности
                recommended_model = self.region_stability[region]['recommended_model']
                
                if recommended_model == "Random Forest":
                    # Обучаем Random Forest
                    result = self.train_random_forest(region_data)
                    if result:
                        self.models[region] = {
                            'model_type': 'Random Forest',
                            'model': result['model'],
                            'scaler': result['scaler'],
                            'features': result['features'],
                            'train_r2': result['train_r2'],
                            'train_mae': result['train_mae']
                        }
                        self.selected_models[region] = 'Random Forest'
                        print(f"    ✅ Обучена Random Forest: R²={result['train_r2']:.3f}")
                
                elif recommended_model == "Prophet" and PROPHET_AVAILABLE:
                    # Обучаем Prophet
                    result = self.train_prophet(region_data)
                    if result:
                        self.models[region] = {
                            'model_type': 'Prophet',
                            'model': result['model'],
                            'train_r2': result['train_r2'],
                            'train_mae': result['train_mae']
                        }
                        self.selected_models[region] = 'Prophet'
                        print(f"    ✅ Обучен Prophet: R²={result['train_r2']:.3f}")
                
                else:
                    print(f"    ⚠️  Prophet недоступен, используем Random Forest")
                    result = self.train_random_forest(region_data)
                    if result:
                        self.models[region] = {
                            'model_type': 'Random Forest',
                            'model': result['model'],
                            'scaler': result['scaler'],
                            'features': result['features'],
                            'train_r2': result['train_r2'],
                            'train_mae': result['train_mae']
                        }
                        self.selected_models[region] = 'Random Forest'
                        print(f"    ✅ Обучена Random Forest (fallback): R²={result['train_r2']:.3f}")
    
    def train_random_forest(self, region_data):
        """Обучение Random Forest модели"""
        try:
            # Подготавливаем признаки
            data = self.prepare_features_for_sklearn(region_data)
            
            # Признаки для обучения
            features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period', 'time_squared', 'time_cubed']
            X = data[features].fillna(0)
            y = data['total_revenue']
            
            # Масштабирование признаков
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Обучение модели
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            # Метрики
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            return {
                'model': model,
                'scaler': scaler,
                'features': features,
                'train_r2': r2,
                'train_mae': mae
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка в Random Forest: {str(e)}")
            return None
    
    def train_prophet(self, region_data):
        """Обучение Prophet модели"""
        try:
            # Подготавливаем данные для Prophet
            prophet_data = self.prepare_features_for_prophet(region_data)
            
            # Создаем и обучаем модель
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            model.fit(prophet_data)
            
            # Предсказание на исторических данных
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            # Метрики
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            return {
                'model': model,
                'train_r2': r2,
                'train_mae': mae
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка в Prophet: {str(e)}")
            return None
    
    def create_future_forecast(self):
        """Создание прогноза на будущие периоды (сентябрь 2025 - декабрь 2026)"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА НА БУДУЩИЕ ПЕРИОДЫ:")
        print(f"  Период: сентябрь 2025 - декабрь 2026")
        
        if not self.models:
            print("❌ Нет обученных моделей")
            return None
        
        forecast_results = {}
        
        for region, model_info in self.models.items():
            print(f"\n  🌍 Прогноз для {region} ({model_info['model_type']}):")
            
            try:
                if model_info['model_type'] == 'Random Forest':
                    # Прогноз Random Forest
                    forecast_data = []
                    
                    # Создаем периоды с сентября 2025 по декабрь 2026
                    for year in [2025, 2026]:
                        start_month = 9 if year == 2025 else 1
                        end_month = 12 if year == 2026 else 12
                        
                        for month in range(start_month, end_month + 1):
                            period_data = {
                                'year': year,
                                'month': month,
                                'time_index': (year - self.historical_data['year'].min()) * 12 + (month - 1),
                                'month_sin': np.sin(2 * np.pi * month / 12),
                                'month_cos': np.cos(2 * np.pi * month / 12),
                                'quarter': ((month - 1) // 3) + 1,
                                'holiday_period': 1 if month in [12, 1, 2, 3, 5] else 0
                            }
                            
                            # Добавляем квартальные признаки
                            for q in range(1, 5):
                                period_data[f'q{q}'] = 1 if period_data['quarter'] == q else 0
                            
                            # Полиномиальные признаки
                            period_data['time_squared'] = period_data['time_index'] ** 2
                            period_data['time_cubed'] = period_data['time_index'] ** 3
                            
                            forecast_data.append(period_data)
                    
                    # Создаем DataFrame для прогноза
                    forecast_df = pd.DataFrame(forecast_data)
                    features = model_info['features']
                    X_forecast = forecast_df[features].fillna(0)
                    
                    # Прогноз
                    X_forecast_scaled = model_info['scaler'].transform(X_forecast)
                    forecast_values = model_info['model'].predict(X_forecast_scaled)
                    
                    # Убираем отрицательные значения
                    forecast_values = np.maximum(forecast_values, 0)
                    
                    forecast_results[region] = {
                        'values': forecast_values,
                        'periods': forecast_df[['year', 'month']].to_dict('records'),
                        'model_type': 'Random Forest'
                    }
                    
                    # Анализ прогноза
                    total_forecast = np.sum(forecast_values)
                    print(f"    Random Forest: {total_forecast:,.0f} ₽")
                    
                    # Анализ по годам
                    forecast_2025 = np.sum([v for v, p in zip(forecast_values, forecast_df[['year', 'month']].to_dict('records')) if p['year'] == 2025])
                    forecast_2026 = np.sum([v for v, p in zip(forecast_values, forecast_df[['year', 'month']].to_dict('records')) if p['year'] == 2026])
                    print(f"      2025 (сент-дек): {forecast_2025:,.0f} ₽")
                    print(f"      2026 (янв-дек): {forecast_2026:,.0f} ₽")
                
                elif model_info['model_type'] == 'Prophet':
                    # Прогноз Prophet
                    prophet_model = model_info['model']
                    
                    # Создаем будущие периоды для Prophet (сентябрь 2025 - декабрь 2026)
                    future_dates = pd.date_range(start='2025-09-01', end='2026-12-01', freq='MS')
                    future_df = pd.DataFrame({'ds': future_dates})
                    
                    # Прогноз
                    forecast = prophet_model.predict(future_df)
                    forecast_values = np.maximum(forecast['yhat'].values, 0)
                    
                    forecast_results[region] = {
                        'values': forecast_values,
                        'periods': [{'year': d.year, 'month': d.month} for d in future_dates],
                        'model_type': 'Prophet'
                    }
                    
                    # Анализ прогноза
                    total_forecast = np.sum(forecast_values)
                    print(f"    Prophet: {total_forecast:,.0f} ₽")
                    
                    # Анализ по годам
                    forecast_2025 = np.sum([v for v, d in zip(forecast_values, future_dates) if d.year == 2025])
                    forecast_2026 = np.sum([v for v, d in zip(forecast_values, future_dates) if d.year == 2026])
                    print(f"      2025 (сент-дек): {forecast_2025:,.0f} ₽")
                    print(f"      2026 (янв-дек): {forecast_2026:,.0f} ₽")
                
            except Exception as e:
                print(f"    ❌ Ошибка в прогнозе: {str(e)}")
                continue
        
        self.our_forecast = forecast_results
        return forecast_results
    
    def compare_with_analyst_forecast(self):
        """Сравнение нашего прогноза с прогнозом аналитиков"""
        print(f"\n📊 СРАВНЕНИЕ С ПРОГНОЗОМ АНАЛИТИКОВ:")
        print(f"  Период сравнения: сентябрь 2025 - август 2026")
        
        if not self.our_forecast or self.analyst_forecast is None:
            print("❌ Нет нашего прогноза или прогноза аналитиков для сравнения")
            return None
        
        comparison_results = {}
        
        for region, our_forecast_info in self.our_forecast.items():
            print(f"\n  🌍 Регион: {region} ({our_forecast_info['model_type']}):")
            
            # Получаем прогноз аналитиков для региона
            analyst_region = self.analyst_forecast[self.analyst_forecast['region_to'] == region].copy()
            
            if len(analyst_region) == 0:
                print(f"    ⚠️  Нет прогноза аналитиков для региона {region}")
                continue
            
            analyst_region = analyst_region.sort_values(['year', 'month'])
            analyst_values = analyst_region['total_revenue'].values
            analyst_total = np.sum(analyst_values)
            
            print(f"    📊 Прогноз аналитиков: {analyst_total:,.0f} ₽")
            
            # Получаем наш прогноз за тот же период (сентябрь 2025 - август 2026)
            our_periods = our_forecast_info['periods']
            our_values = our_forecast_info['values']
            
            # Фильтруем наш прогноз за период сентябрь 2025 - август 2026
            our_filtered_values = []
            for i, period in enumerate(our_periods):
                if (period['year'] == 2025 and period['month'] >= 9) or (period['year'] == 2026 and period['month'] <= 8):
                    our_filtered_values.append(our_values[i])
            
            if len(our_filtered_values) != len(analyst_values):
                print(f"    ⚠️  Несоответствие размеров данных")
                continue
            
            our_total = np.sum(our_filtered_values)
            
            # Вычисляем метрики сравнения
            mae = mean_absolute_error(analyst_values, our_filtered_values)
            rmse = np.sqrt(mean_squared_error(analyst_values, our_filtered_values))
            mape = np.mean(np.abs((analyst_values - our_filtered_values) / analyst_values)) * 100
            
            # Общая ошибка
            total_error = our_total - analyst_total
            total_error_pct = (total_error / analyst_total * 100) if analyst_total > 0 else 0
            
            comparison_results[region] = {
                'model_type': our_forecast_info['model_type'],
                'our_forecast': our_total,
                'analyst_forecast': analyst_total,
                'total_error': total_error,
                'total_error_pct': total_error_pct,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            print(f"    Наш прогноз: {our_total:,.0f} ₽")
            print(f"    Ошибка: {total_error:,.0f} ₽ ({total_error_pct:+.1f}%)")
            print(f"    MAPE: {mape:.1f}%")
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def summarize_future_forecast(self):
        """Сводка прогноза на будущие периоды"""
        print(f"\n📈 СВОДКА ПРОГНОЗА НА БУДУЩИЕ ПЕРИОДЫ:")
        
        if not self.our_forecast:
            print("❌ Нет прогноза для сводки")
            return None
        
        # Общий прогноз
        total_forecast = 0
        forecast_2025 = 0
        forecast_2026 = 0
        
        for region, forecast_info in self.our_forecast.items():
            region_total = np.sum(forecast_info['values'])
            total_forecast += region_total
            
            # Анализ по годам
            for i, period in enumerate(forecast_info['periods']):
                if period['year'] == 2025:
                    forecast_2025 += forecast_info['values'][i]
                elif period['year'] == 2026:
                    forecast_2026 += forecast_info['values'][i]
        
        print(f"\n  📊 ОБЩИЙ ПРОГНОЗ:")
        print(f"    Общая выручка: {total_forecast:,.0f} ₽")
        print(f"    2025 (сент-дек): {forecast_2025:,.0f} ₽")
        print(f"    2026 (янв-дек): {forecast_2026:,.0f} ₽")
        
        # Анализ по регионам
        print(f"\n  🌍 ПРОГНОЗ ПО РЕГИОНАМ:")
        region_totals = {}
        for region, forecast_info in self.our_forecast.items():
            region_total = np.sum(forecast_info['values'])
            region_totals[region] = region_total
        
        # Сортируем по убыванию
        sorted_regions = sorted(region_totals.items(), key=lambda x: x[1], reverse=True)
        
        for region, total in sorted_regions:
            percentage = (total / total_forecast * 100) if total_forecast > 0 else 0
            print(f"    {region}: {total:,.0f} ₽ ({percentage:.1f}%)")
        
        # Сравнение с аналитиками
        if self.comparison_results:
            print(f"\n  📊 СРАВНЕНИЕ С АНАЛИТИКАМИ:")
            total_analyst = sum([r['analyst_forecast'] for r in self.comparison_results.values()])
            total_our = sum([r['our_forecast'] for r in self.comparison_results.values()])
            total_diff = total_our - total_analyst
            total_diff_pct = (total_diff / total_analyst * 100) if total_analyst > 0 else 0
            
            print(f"    Прогноз аналитиков: {total_analyst:,.0f} ₽")
            print(f"    Наш прогноз: {total_our:,.0f} ₽")
            print(f"    Разница: {total_diff:,.0f} ₽ ({total_diff_pct:+.1f}%)")
        
        return self.our_forecast
    
    def save_future_forecast(self, output_file='Future_Forecast_Results.csv'):
        """Сохранение прогноза на будущие периоды"""
        if not self.our_forecast:
            print("❌ Нет прогноза для сохранения")
            return None
        
        # Собираем все прогнозы
        all_forecasts = []
        
        for region, forecast_info in self.our_forecast.items():
            for i, (period, value) in enumerate(zip(forecast_info['periods'], forecast_info['values'])):
                all_forecasts.append({
                    'region': region,
                    'model_type': forecast_info['model_type'],
                    'year': period['year'],
                    'month': period['month'],
                    'forecast_revenue': value
                })
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(all_forecasts)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Прогноз на будущие периоды сохранен в файл: {output_file}")
        
        return results_df

def main():
    """Основная функция для прогноза на будущие периоды"""
    print("🔬 ПРОГНОЗ НА БУДУЩИЕ ПЕРИОДЫ")
    print("="*60)
    print("Период: сентябрь 2025 - декабрь 2026")
    print("Сравнение с прогнозом аналитиков")
    print("="*60)
    
    # Инициализация
    future_forecast = FutureForecastHybrid('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и очистка данных
    future_forecast.load_and_clean_data()
    
    # Подготовка агрегированных данных
    future_forecast.prepare_aggregated_data()
    
    # Разделение данных
    future_forecast.split_historical_and_forecast_data()
    
    # Анализ стабильности регионов
    future_forecast.analyze_region_stability()
    
    # Обучение гибридных моделей
    future_forecast.train_hybrid_models()
    
    # Создание прогноза на будущие периоды
    future_forecast.create_future_forecast()
    
    # Сравнение с прогнозом аналитиков
    future_forecast.compare_with_analyst_forecast()
    
    # Сводка прогноза
    future_forecast.summarize_future_forecast()
    
    # Сохранение результатов
    future_forecast.save_future_forecast()
    
    print(f"\n🎉 Прогноз на будущие периоды завершен!")

if __name__ == "__main__":
    main()
