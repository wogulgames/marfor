#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение различных моделей прогнозирования
Включает: Linear Regression, Ridge, Random Forest, XGBoost, Prophet
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

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost не установлен. Установите: pip install xgboost")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet не установлен. Установите: pip install prophet")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelComparison:
    def __init__(self, csv_file=None):
        """Инициализация сравнения моделей"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.models = {}
        self.results = {}
        self.forecast_results = {}
        
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
    
    def prepare_aggregated_data(self, train_end_year=2025, train_end_month=9):
        """Подготовка агрегированных данных для обучения"""
        print(f"\n📚 ПОДГОТОВКА АГРЕГИРОВАННЫХ ДАННЫХ:")
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
            
            return self.aggregated_df
        else:
            print("❌ Не найдены необходимые колонки")
            return None
    
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
        
        # Добавляем XGBoost если доступен
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        results = {}
        
        for name, model in models.items():
            try:
                # Обучение
                if name == 'XGBoost':
                    model.fit(X, y)  # XGBoost не требует масштабирования
                    y_pred = model.predict(X)
                else:
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                
                # Метрики
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                results[name] = {
                    'model': model,
                    'scaler': scaler if name != 'XGBoost' else None,
                    'features': features,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': y_pred
                }
                
                print(f"    {name}: R²={r2:.3f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}")
                
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
            
            # Метрики
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            print(f"    Prophet: R²={r2:.3f}, MAE={mae:,.0f}, RMSE={rmse:,.0f}")
            
            return {
                'model': model,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"    ❌ Ошибка в Prophet: {str(e)}")
            return None
    
    def train_all_models(self):
        """Обучение всех моделей для всех регионов"""
        print(f"\n🤖 ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ:")
        
        if self.aggregated_df is None:
            print("❌ Нет агрегированных данных для обучения")
            return None
        
        for region in self.aggregated_df['region_to'].unique():
            region_data = self.aggregated_df[self.aggregated_df['region_to'] == region].copy()
            
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
    
    def create_forecasts(self, forecast_periods=4):
        """Создание прогнозов всеми моделями"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗОВ НА {forecast_periods} ПЕРИОДА:")
        
        if not self.models:
            print("❌ Нет обученных моделей")
            return None
        
        forecast_results = {}
        
        for region, region_models in self.models.items():
            print(f"\n  🌍 Прогноз для {region}:")
            
            region_forecasts = {}
            
            # Прогнозы scikit-learn моделей
            if 'sklearn' in region_models:
                for model_name, model_info in region_models['sklearn'].items():
                    try:
                        # Создаем будущие периоды
                        last_data = self.aggregated_df[self.aggregated_df['region_to'] == region].sort_values(['year', 'month']).iloc[-1]
                        last_year = last_data['year']
                        last_month = last_data['month']
                        
                        # Создаем прогнозные периоды
                        forecast_data = []
                        for i in range(1, forecast_periods + 1):
                            month = last_month + i
                            year = last_year
                            while month > 12:
                                month -= 12
                                year += 1
                            
                            period_data = {
                                'year': year,
                                'month': month,
                                'time_index': (year - self.aggregated_df['year'].min()) * 12 + (month - 1),
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
                        if model_name == 'XGBoost':
                            forecast_values = model_info['model'].predict(X_forecast)
                        else:
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
                    
                    # Создаем будущие периоды для Prophet
                    last_data = self.aggregated_df[self.aggregated_df['region_to'] == region].sort_values(['year', 'month']).iloc[-1]
                    last_date = pd.to_datetime(f"{last_data['year']}-{last_data['month']:02d}-01")
                    
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
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
        
        self.forecast_results = forecast_results
        return forecast_results
    
    def compare_models(self):
        """Сравнение качества моделей"""
        print(f"\n📊 СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ:")
        
        if not self.models:
            print("❌ Нет обученных моделей для сравнения")
            return None
        
        # Собираем метрики всех моделей
        all_metrics = []
        
        for region, region_models in self.models.items():
            # Scikit-learn модели
            if 'sklearn' in region_models:
                for model_name, model_info in region_models['sklearn'].items():
                    all_metrics.append({
                        'region': region,
                        'model': model_name,
                        'r2': model_info['r2'],
                        'mae': model_info['mae'],
                        'rmse': model_info['rmse']
                    })
            
            # Prophet модель
            if 'prophet' in region_models and region_models['prophet'] is not None:
                prophet_info = region_models['prophet']
                all_metrics.append({
                    'region': region,
                    'model': 'Prophet',
                    'r2': prophet_info['r2'],
                    'mae': prophet_info['mae'],
                    'rmse': prophet_info['rmse']
                })
        
        # Создаем DataFrame для анализа
        metrics_df = pd.DataFrame(all_metrics)
        
        if len(metrics_df) == 0:
            print("❌ Нет метрик для сравнения")
            return None
        
        # Анализ по моделям
        print(f"\n  📈 СРЕДНИЕ МЕТРИКИ ПО МОДЕЛЯМ:")
        model_summary = metrics_df.groupby('model').agg({
            'r2': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std']
        }).round(3)
        
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            avg_r2 = model_data['r2'].mean()
            avg_mae = model_data['mae'].mean()
            avg_rmse = model_data['rmse'].mean()
            
            print(f"    {model}:")
            print(f"      R²: {avg_r2:.3f} ± {model_data['r2'].std():.3f}")
            print(f"      MAE: {avg_mae:,.0f} ± {model_data['mae'].std():,.0f}")
            print(f"      RMSE: {avg_rmse:,.0f} ± {model_data['rmse'].std():,.0f}")
        
        # Лучшие модели по регионам
        print(f"\n  🏆 ЛУЧШИЕ МОДЕЛИ ПО РЕГИОНАМ:")
        for region in metrics_df['region'].unique():
            region_data = metrics_df[metrics_df['region'] == region]
            best_model = region_data.loc[region_data['r2'].idxmax()]
            print(f"    {region}: {best_model['model']} (R²={best_model['r2']:.3f})")
        
        return metrics_df
    
    def save_comparison_results(self, output_file='Model_Comparison_Results.csv'):
        """Сохранение результатов сравнения"""
        if not self.forecast_results:
            print("❌ Нет результатов прогноза для сохранения")
            return None
        
        # Собираем все прогнозы
        all_forecasts = []
        
        for region, region_forecasts in self.forecast_results.items():
            for model_name, model_forecast in region_forecasts.items():
                for i, (period, value) in enumerate(zip(model_forecast['periods'], model_forecast['values'])):
                    all_forecasts.append({
                        'region': region,
                        'model': model_name,
                        'year': period['year'],
                        'month': period['month'],
                        'forecast_revenue': value
                    })
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(all_forecasts)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Результаты сравнения моделей сохранены в файл: {output_file}")
        
        return results_df

def main():
    """Основная функция для сравнения моделей"""
    print("🔬 СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
    print("="*60)
    print("Модели: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, Prophet")
    print("="*60)
    
    # Инициализация
    comparator = ModelComparison('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и очистка данных
    comparator.load_and_clean_data()
    
    # Подготовка агрегированных данных
    comparator.prepare_aggregated_data()
    
    # Обучение всех моделей
    comparator.train_all_models()
    
    # Создание прогнозов
    comparator.create_forecasts()
    
    # Сравнение моделей
    comparator.compare_models()
    
    # Сохранение результатов
    comparator.save_comparison_results()
    
    print(f"\n🎉 Сравнение моделей завершено!")

if __name__ == "__main__":
    main()
