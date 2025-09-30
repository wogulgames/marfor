#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Валидация моделей на периоде июнь-август 2025
Сравнение прогнозов с фактическими данными
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

class ModelValidation:
    def __init__(self, csv_file=None):
        """Инициализация валидации моделей"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.train_data = None
        self.test_data = None
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
    
    def train_sklearn_models(self, region_data):
        """Обучение scikit-learn моделей"""
        print(f"  🔧 Обучение scikit-learn моделей...")
        
        # Подготавливаем признаки
        data = self.prepare_features_for_sklearn(region_data)
        
        # Признаки для обучения
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period', 'time_squared', 'time_cubed']
        X = data[features].fillna(0)
        y = data['total_revenue']
        
        # Масштабирование признаков
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Обучение
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                
                # Метрики на обучающих данных
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'features': features,
                    'train_r2': r2,
                    'train_mae': mae,
                    'train_rmse': rmse
                }
                
                print(f"    {name}: Train R²={r2:.3f}, MAE={mae:,.0f}")
                
            except Exception as e:
                print(f"    ❌ Ошибка в {name}: {str(e)}")
                continue
        
        return results
    
    def train_prophet_model(self, region_data):
        """Обучение Prophet модели"""
        if not PROPHET_AVAILABLE:
            print("  ⚠️  Prophet недоступен")
            return None
        
        print(f"  🔧 Обучение Prophet модели...")
        
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
            
            # Метрики на обучающих данных
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            print(f"    Prophet: Train R²={r2:.3f}, MAE={mae:,.0f}")
            
            return {
                'model': model,
                'train_r2': r2,
                'train_mae': mae,
                'train_rmse': rmse
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка в Prophet: {str(e)}")
            return None
    
    def train_all_models(self):
        """Обучение всех моделей для всех регионов"""
        print(f"\n🤖 ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ:")
        
        if self.train_data is None:
            print("❌ Нет обучающих данных")
            return None
        
        for region in self.train_data['region_to'].unique():
            region_data = self.train_data[self.train_data['region_to'] == region].copy()
            
            if len(region_data) > 6:  # Минимум 6 месяцев данных
                print(f"\n  🌍 Регион: {region}")
                
                # Scikit-learn модели
                sklearn_results = self.train_sklearn_models(region_data)
                
                # Prophet модель
                prophet_result = self.train_prophet_model(region_data)
                
                # Сохраняем результаты
                self.models[region] = {
                    'sklearn': sklearn_results,
                    'prophet': prophet_result
                }
    
    def create_forecasts(self):
        """Создание прогнозов на тестовый период"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗОВ НА ТЕСТОВЫЙ ПЕРИОД:")
        
        if not self.models or self.test_data is None:
            print("❌ Нет обученных моделей или тестовых данных")
            return None
        
        forecast_results = {}
        
        for region, region_models in self.models.items():
            print(f"\n  🌍 Прогноз для {region}:")
            
            region_forecasts = {}
            
            # Прогнозы scikit-learn моделей
            if 'sklearn' in region_models:
                for model_name, model_info in region_models['sklearn'].items():
                    try:
                        # Создаем прогнозные периоды (июнь-август 2025)
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
                        
                        region_forecasts[model_name] = {
                            'values': forecast_values,
                            'periods': forecast_df[['year', 'month']].to_dict('records')
                        }
                        
                        total_forecast = np.sum(forecast_values)
                        print(f"    {model_name}: {total_forecast:,.0f} ₽")
                        
                    except Exception as e:
                        print(f"    ❌ Ошибка в {model_name}: {str(e)}")
                        continue
            
            # Прогноз Prophet
            if 'prophet' in region_models and region_models['prophet'] is not None:
                try:
                    prophet_model = region_models['prophet']['model']
                    
                    # Создаем будущие периоды для Prophet (июнь-август 2025)
                    future_dates = pd.date_range(start='2025-06-01', end='2025-08-01', freq='MS')
                    future_df = pd.DataFrame({'ds': future_dates})
                    
                    # Прогноз
                    forecast = prophet_model.predict(future_df)
                    forecast_values = np.maximum(forecast['yhat'].values, 0)
                    
                    region_forecasts['Prophet'] = {
                        'values': forecast_values,
                        'periods': [{'year': d.year, 'month': d.month} for d in future_dates]
                    }
                    
                    total_forecast = np.sum(forecast_values)
                    print(f"    Prophet: {total_forecast:,.0f} ₽")
                    
                except Exception as e:
                    print(f"    ❌ Ошибка в Prophet: {str(e)}")
                    continue
            
            forecast_results[region] = region_forecasts
        
        self.forecasts = forecast_results
        return forecast_results
    
    def compare_with_actual(self):
        """Сравнение прогнозов с фактическими данными"""
        print(f"\n📊 СРАВНЕНИЕ ПРОГНОЗОВ С ФАКТИЧЕСКИМИ ДАННЫМИ:")
        
        if not self.forecasts or self.test_data is None:
            print("❌ Нет прогнозов или тестовых данных для сравнения")
            return None
        
        comparison_results = {}
        
        for region, region_forecasts in self.forecasts.items():
            print(f"\n  🌍 Регион: {region}")
            
            # Получаем фактические данные для региона
            region_actual = self.test_data[self.test_data['region_to'] == region].copy()
            
            if len(region_actual) == 0:
                print(f"    ⚠️  Нет фактических данных для региона {region}")
                continue
            
            region_actual = region_actual.sort_values(['year', 'month'])
            actual_values = region_actual['total_revenue'].values
            actual_total = np.sum(actual_values)
            
            print(f"    📊 Фактическая выручка: {actual_total:,.0f} ₽")
            
            region_comparison = {}
            
            # Сравниваем с каждой моделью
            for model_name, model_forecast in region_forecasts.items():
                forecast_values = model_forecast['values']
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
                    
                    region_comparison[model_name] = {
                        'forecast_total': forecast_total,
                        'actual_total': actual_total,
                        'total_error': total_error,
                        'total_error_pct': total_error_pct,
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape
                    }
                    
                    print(f"    {model_name}:")
                    print(f"      Прогноз: {forecast_total:,.0f} ₽")
                    print(f"      Ошибка: {total_error:,.0f} ₽ ({total_error_pct:+.1f}%)")
                    print(f"      MAPE: {mape:.1f}%")
                else:
                    print(f"    ⚠️  {model_name}: Несоответствие размеров данных")
            
            comparison_results[region] = region_comparison
        
        self.validation_results = comparison_results
        return comparison_results
    
    def summarize_results(self):
        """Сводка результатов валидации"""
        print(f"\n📈 СВОДКА РЕЗУЛЬТАТОВ ВАЛИДАЦИИ:")
        
        if not self.validation_results:
            print("❌ Нет результатов для сводки")
            return None
        
        # Собираем все метрики
        all_metrics = []
        
        for region, region_results in self.validation_results.items():
            for model_name, metrics in region_results.items():
                all_metrics.append({
                    'region': region,
                    'model': model_name,
                    'mape': metrics['mape'],
                    'total_error_pct': metrics['total_error_pct'],
                    'mae': metrics['mae']
                })
        
        if len(all_metrics) == 0:
            print("❌ Нет метрик для анализа")
            return None
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Анализ по моделям
        print(f"\n  📊 СРЕДНИЕ МЕТРИКИ ПО МОДЕЛЯМ:")
        
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            avg_mape = model_data['mape'].mean()
            avg_error_pct = model_data['total_error_pct'].mean()
            avg_mae = model_data['mae'].mean()
            
            print(f"    {model}:")
            print(f"      Средний MAPE: {avg_mape:.1f}%")
            print(f"      Средняя ошибка: {avg_error_pct:+.1f}%")
            print(f"      Средний MAE: {avg_mae:,.0f} ₽")
        
        # Лучшие модели по регионам
        print(f"\n  🏆 ЛУЧШИЕ МОДЕЛИ ПО РЕГИОНАМ (по MAPE):")
        for region in metrics_df['region'].unique():
            region_data = metrics_df[metrics_df['region'] == region]
            best_model = region_data.loc[region_data['mape'].idxmin()]
            print(f"    {region}: {best_model['model']} (MAPE={best_model['mape']:.1f}%)")
        
        # Общий рейтинг моделей
        print(f"\n  🥇 ОБЩИЙ РЕЙТИНГ МОДЕЛЕЙ (по среднему MAPE):")
        model_ranking = metrics_df.groupby('model')['mape'].mean().sort_values()
        for i, (model, mape) in enumerate(model_ranking.items(), 1):
            print(f"    {i}. {model}: MAPE={mape:.1f}%")
        
        return metrics_df
    
    def save_results(self, output_file='Model_Validation_Results.csv'):
        """Сохранение результатов валидации"""
        if not self.validation_results:
            print("❌ Нет результатов для сохранения")
            return None
        
        # Собираем все результаты
        all_results = []
        
        for region, region_results in self.validation_results.items():
            for model_name, metrics in region_results.items():
                all_results.append({
                    'region': region,
                    'model': model_name,
                    'forecast_total': metrics['forecast_total'],
                    'actual_total': metrics['actual_total'],
                    'total_error': metrics['total_error'],
                    'total_error_pct': metrics['total_error_pct'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape']
                })
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Результаты валидации сохранены в файл: {output_file}")
        
        return results_df

def main():
    """Основная функция для валидации моделей"""
    print("🔬 ВАЛИДАЦИЯ МОДЕЛЕЙ НА ПЕРИОДЕ ИЮНЬ-АВГУСТ 2025")
    print("="*60)
    print("Обучающие данные: до мая 2025")
    print("Тестовые данные: июнь-август 2025")
    print("="*60)
    
    # Инициализация
    validator = ModelValidation('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и очистка данных
    validator.load_and_clean_data()
    
    # Подготовка агрегированных данных
    validator.prepare_aggregated_data()
    
    # Разделение данных
    validator.split_data()
    
    # Обучение всех моделей
    validator.train_all_models()
    
    # Создание прогнозов
    validator.create_forecasts()
    
    # Сравнение с фактическими данными
    validator.compare_with_actual()
    
    # Сводка результатов
    validator.summarize_results()
    
    # Сохранение результатов
    validator.save_results()
    
    print(f"\n🎉 Валидация моделей завершена!")

if __name__ == "__main__":
    main()
