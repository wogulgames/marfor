#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гибридная каскадная модель с автоматическим выбором модели
Random Forest для стабильных регионов, Prophet для нестабильных
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

class HybridCascadedModel:
    def __init__(self, csv_file=None):
        """Инициализация гибридной каскадной модели"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.train_data = None
        self.test_data = None
        self.region_stability = {}
        self.selected_models = {}
        self.models = {}
        self.forecasts = {}
        self.validation_results = {}
        
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
    
    def split_data(self, train_end_year=2025, train_end_month=5, test_start_year=2025, test_start_month=6, test_end_month=8):
        """Разделение данных на обучающую и тестовую выборки"""
        print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"  Обучающие данные: до {train_end_year}.{train_end_month:02d}")
        print(f"  Тестовые данные: {test_start_year}.{test_start_month:02d} - {test_start_year}.{test_end_month:02d}")
        
        # Обучающие данные (до мая 2025)
        train_mask = (
            (self.aggregated_df['year'] < train_end_year) | 
            ((self.aggregated_df['year'] == train_end_year) & (self.aggregated_df['month'] <= train_end_month))
        )
        self.train_data = self.aggregated_df[train_mask].copy()
        
        # Тестовые данные (июнь-август 2025)
        test_mask = (
            (self.aggregated_df['year'] == test_start_year) & 
            (self.aggregated_df['month'] >= test_start_month) & 
            (self.aggregated_df['month'] <= test_end_month)
        )
        self.test_data = self.aggregated_df[test_mask].copy()
        
        print(f"  Обучающих записей: {len(self.train_data)}")
        print(f"  Тестовых записей: {len(self.test_data)}")
        
        # Анализ обучающих данных
        train_revenue = self.train_data['total_revenue'].sum()
        print(f"  Выручка в обучающих данных: {train_revenue:,.0f} ₽")
        
        # Анализ тестовых данных
        if len(self.test_data) > 0:
            test_revenue = self.test_data['total_revenue'].sum()
            print(f"  Выручка в тестовых данных: {test_revenue:,.0f} ₽")
        else:
            print("  ⚠️  Нет тестовых данных за указанный период")
    
    def analyze_region_stability(self):
        """Анализ стабильности данных по регионам"""
        print(f"\n🔍 АНАЛИЗ СТАБИЛЬНОСТИ ДАННЫХ ПО РЕГИОНАМ:")
        
        if self.train_data is None:
            print("❌ Нет обучающих данных для анализа")
            return None
        
        for region in self.train_data['region_to'].unique():
            region_data = self.train_data[self.train_data['region_to'] == region].copy()
            
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
        
        if self.train_data is None or not self.region_stability:
            print("❌ Нет обучающих данных или анализа стабильности")
            return None
        
        for region in self.train_data['region_to'].unique():
            region_data = self.train_data[self.train_data['region_to'] == region].copy()
            
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
    
    def create_hybrid_forecasts(self):
        """Создание прогнозов гибридной моделью"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗОВ ГИБРИДНОЙ МОДЕЛЬЮ:")
        
        if not self.models or self.test_data is None:
            print("❌ Нет обученных моделей или тестовых данных")
            return None
        
        forecast_results = {}
        
        for region, model_info in self.models.items():
            print(f"\n  🌍 Прогноз для {region} ({model_info['model_type']}):")
            
            try:
                if model_info['model_type'] == 'Random Forest':
                    # Прогноз Random Forest
                    forecast_data = []
                    for month in [6, 7, 8]:  # июнь, июль, август
                        period_data = {
                            'year': 2025,
                            'month': month,
                            'time_index': (2025 - self.train_data['year'].min()) * 12 + (month - 1),
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
                    
                    total_forecast = np.sum(forecast_values)
                    print(f"    Random Forest: {total_forecast:,.0f} ₽")
                
                elif model_info['model_type'] == 'Prophet':
                    # Прогноз Prophet
                    prophet_model = model_info['model']
                    
                    # Создаем будущие периоды для Prophet (июнь-август 2025)
                    future_dates = pd.date_range(start='2025-06-01', end='2025-08-01', freq='MS')
                    future_df = pd.DataFrame({'ds': future_dates})
                    
                    # Прогноз
                    forecast = prophet_model.predict(future_df)
                    forecast_values = np.maximum(forecast['yhat'].values, 0)
                    
                    forecast_results[region] = {
                        'values': forecast_values,
                        'periods': [{'year': d.year, 'month': d.month} for d in future_dates],
                        'model_type': 'Prophet'
                    }
                    
                    total_forecast = np.sum(forecast_values)
                    print(f"    Prophet: {total_forecast:,.0f} ₽")
                
            except Exception as e:
                print(f"    ❌ Ошибка в прогнозе: {str(e)}")
                continue
        
        self.forecasts = forecast_results
        return forecast_results
    
    def compare_with_actual(self):
        """Сравнение прогнозов с фактическими данными"""
        print(f"\n📊 СРАВНЕНИЕ ГИБРИДНЫХ ПРОГНОЗОВ С ФАКТИЧЕСКИМИ ДАННЫМИ:")
        
        if not self.forecasts or self.test_data is None:
            print("❌ Нет прогнозов или тестовых данных для сравнения")
            return None
        
        comparison_results = {}
        
        for region, forecast_info in self.forecasts.items():
            print(f"\n  🌍 Регион: {region} ({forecast_info['model_type']}):")
            
            # Получаем фактические данные для региона
            region_actual = self.test_data[self.test_data['region_to'] == region].copy()
            
            if len(region_actual) == 0:
                print(f"    ⚠️  Нет фактических данных для региона {region}")
                continue
            
            region_actual = region_actual.sort_values(['year', 'month'])
            actual_values = region_actual['total_revenue'].values
            actual_total = np.sum(actual_values)
            
            print(f"    📊 Фактическая выручка: {actual_total:,.0f} ₽")
            
            # Сравниваем с прогнозом
            forecast_values = forecast_info['values']
            forecast_total = np.sum(forecast_values)
            
            # Вычисляем метрики
            if len(forecast_values) == len(actual_values):
                # Поэлементное сравнение
                mae = mean_absolute_error(actual_values, forecast_values)
                rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
                mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
                
                # Общая ошибка
                total_error = forecast_total - actual_total
                total_error_pct = (total_error / actual_total * 100) if actual_total > 0 else 0
                
                comparison_results[region] = {
                    'model_type': forecast_info['model_type'],
                    'forecast_total': forecast_total,
                    'actual_total': actual_total,
                    'total_error': total_error,
                    'total_error_pct': total_error_pct,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }
                
                print(f"    Прогноз: {forecast_total:,.0f} ₽")
                print(f"    Ошибка: {total_error:,.0f} ₽ ({total_error_pct:+.1f}%)")
                print(f"    MAPE: {mape:.1f}%")
            else:
                print(f"    ⚠️  Несоответствие размеров данных")
        
        self.validation_results = comparison_results
        return comparison_results
    
    def summarize_hybrid_results(self):
        """Сводка результатов гибридной модели"""
        print(f"\n📈 СВОДКА РЕЗУЛЬТАТОВ ГИБРИДНОЙ МОДЕЛИ:")
        
        if not self.validation_results:
            print("❌ Нет результатов для сводки")
            return None
        
        # Анализ по типам моделей
        model_types = {}
        for region, results in self.validation_results.items():
            model_type = results['model_type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(results)
        
        print(f"\n  📊 РЕЗУЛЬТАТЫ ПО ТИПАМ МОДЕЛЕЙ:")
        
        for model_type, results_list in model_types.items():
            avg_mape = np.mean([r['mape'] for r in results_list])
            avg_error_pct = np.mean([r['total_error_pct'] for r in results_list])
            avg_mae = np.mean([r['mae'] for r in results_list])
            
            print(f"    {model_type}:")
            print(f"      Количество регионов: {len(results_list)}")
            print(f"      Средний MAPE: {avg_mape:.1f}%")
            print(f"      Средняя ошибка: {avg_error_pct:+.1f}%")
            print(f"      Средний MAE: {avg_mae:,.0f} ₽")
        
        # Общие метрики
        all_mape = [r['mape'] for r in self.validation_results.values()]
        all_error_pct = [r['total_error_pct'] for r in self.validation_results.values()]
        all_mae = [r['mae'] for r in self.validation_results.values()]
        
        print(f"\n  📊 ОБЩИЕ МЕТРИКИ ГИБРИДНОЙ МОДЕЛИ:")
        print(f"    Средний MAPE: {np.mean(all_mape):.1f}%")
        print(f"    Средняя ошибка: {np.mean(all_error_pct):+.1f}%")
        print(f"    Средний MAE: {np.mean(all_mae):,.0f} ₽")
        
        # Лучшие результаты
        best_region = min(self.validation_results.items(), key=lambda x: x[1]['mape'])
        print(f"\n  🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"    Регион: {best_region[0]} ({best_region[1]['model_type']})")
        print(f"    MAPE: {best_region[1]['mape']:.1f}%")
        
        return self.validation_results
    
    def save_hybrid_results(self, output_file='Hybrid_Model_Results.csv'):
        """Сохранение результатов гибридной модели"""
        if not self.validation_results:
            print("❌ Нет результатов для сохранения")
            return None
        
        # Собираем все результаты
        all_results = []
        
        for region, results in self.validation_results.items():
            all_results.append({
                'region': region,
                'model_type': results['model_type'],
                'forecast_total': results['forecast_total'],
                'actual_total': results['actual_total'],
                'total_error': results['total_error'],
                'total_error_pct': results['total_error_pct'],
                'mae': results['mae'],
                'rmse': results['rmse'],
                'mape': results['mape']
            })
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Результаты гибридной модели сохранены в файл: {output_file}")
        
        return results_df

def main():
    """Основная функция для гибридной каскадной модели"""
    print("🔬 ГИБРИДНАЯ КАСКАДНАЯ МОДЕЛЬ")
    print("="*60)
    print("Random Forest для стабильных регионов")
    print("Prophet для нестабильных регионов")
    print("Валидация на периоде июнь-август 2025")
    print("="*60)
    
    # Инициализация
    hybrid_model = HybridCascadedModel('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и очистка данных
    hybrid_model.load_and_clean_data()
    
    # Подготовка агрегированных данных
    hybrid_model.prepare_aggregated_data()
    
    # Разделение данных
    hybrid_model.split_data()
    
    # Анализ стабильности регионов
    hybrid_model.analyze_region_stability()
    
    # Обучение гибридных моделей
    hybrid_model.train_hybrid_models()
    
    # Создание прогнозов
    hybrid_model.create_hybrid_forecasts()
    
    # Сравнение с фактическими данными
    hybrid_model.compare_with_actual()
    
    # Сводка результатов
    hybrid_model.summarize_hybrid_results()
    
    # Сохранение результатов
    hybrid_model.save_hybrid_results()
    
    print(f"\n🎉 Гибридная каскадная модель завершена!")

if __name__ == "__main__":
    main()
