#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARFOR - Маркетинговый инструмент прогнозирования
Веб-приложение с тремя модулями: трендовый прогноз, анализ зависимостей, моделирование
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Flask
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime

# Scikit-learn модели
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
app.secret_key = 'marfor-secret-key-2024'

# Настройки загрузки файлов
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MARFORWebApp:
    def __init__(self):
        """Инициализация MARFOR веб-приложения"""
        self.df = None
        self.session_id = None
        self.trend_results = {}
        self.dependency_results = {}
        self.simulation_results = {}
        
    def load_data_from_file(self, file_path: str):
        """Загрузка данных из файла"""
        try:
            if file_path.endswith('.csv'):
                # Пробуем разные разделители
                separators = [',', ';', '\t', '|']
                for sep in separators:
                    try:
                        self.df = pd.read_csv(file_path, sep=sep)
                        if len(self.df.columns) > 1:
                            break
                    except:
                        continue
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            
            if self.df is None or len(self.df.columns) <= 1:
                return False, "Не удалось загрузить файл с правильным разделителем"
            
            # Очистка данных
            self._clean_data()
            return True, f"Загружено {len(self.df)} записей, {len(self.df.columns)} колонок"
            
        except Exception as e:
            return False, f"Ошибка при загрузке файла: {str(e)}"
    
    def _clean_data(self):
        """Очистка данных"""
        initial_count = len(self.df)
        
        # Очистка числовых колонок
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Пробуем преобразовать в числовой формат
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        
        # Удаляем строки где все значения NaN
        self.df = self.df.dropna(how='all')
        
        print(f"Очистка данных: удалено {initial_count - len(self.df)} записей")
    
    def get_data_info(self):
        """Получение информации о данных"""
        if self.df is None:
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'sample_data': self.df.head(5).to_dict('records'),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return info

    # МОДУЛЬ 1: ТРЕНДОВЫЙ ПРОГНОЗ
    def run_trend_forecast(self, config):
        """Запуск трендового прогноза"""
        try:
            target_metric = config['target_metric']
            periods = config['periods']
            model_type = config['model']
            enable_cascade = config['enable_cascade']
            cascade_level = config['cascade_level']
            
            if target_metric not in self.df.columns:
                return False, f"Метрика {target_metric} не найдена в данных"
            
            # Подготавливаем данные для обучения (до августа 2025)
            train_data = self._prepare_training_data()
            
            if len(train_data) < 10:
                return False, "Недостаточно данных для обучения"
            
            # Выбираем модель
            if model_type == 'hybrid':
                model_result = self._run_hybrid_model(train_data, target_metric, periods, enable_cascade, cascade_level)
            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                model_result = self._run_prophet_model(train_data, target_metric, periods)
            elif model_type == 'random_forest':
                model_result = self._run_random_forest_model(train_data, target_metric, periods)
            else:
                model_result = self._run_linear_model(train_data, target_metric, periods)
            
            if model_result['success']:
                self.trend_results = model_result
                return True, "Трендовый прогноз успешно выполнен"
            else:
                return False, model_result['message']
                
        except Exception as e:
            return False, f"Ошибка в трендовом прогнозе: {str(e)}"
    
    def _prepare_training_data(self):
        """Подготовка данных для обучения"""
        if 'year' in self.df.columns and 'month' in self.df.columns:
            # Используем данные до августа 2025 для обучения
            train_mask = (
                (self.df['year'] < 2025) | 
                ((self.df['year'] == 2025) & (self.df['month'] <= 8))
            )
            return self.df[train_mask].copy()
        else:
            # Если временные поля не настроены, используем 80% данных
            train_size = int(len(self.df) * 0.8)
            return self.df.head(train_size).copy()
    
    def _run_hybrid_model(self, train_data, target_metric, periods, enable_cascade, cascade_level):
        """Запуск гибридной модели"""
        try:
            # Агрегируем данные по уровню каскада
            if enable_cascade and cascade_level in train_data.columns:
                aggregated_data = train_data.groupby(['year', 'month', cascade_level])[target_metric].sum().reset_index()
                aggregated_data = aggregated_data.groupby(['year', 'month'])[target_metric].mean().reset_index()
            else:
                aggregated_data = train_data.groupby(['year', 'month'])[target_metric].sum().reset_index()
            
            # Создаем временной индекс
            aggregated_data['time_index'] = (aggregated_data['year'] - aggregated_data['year'].min()) * 12 + (aggregated_data['month'] - 1)
            
            # Анализируем стабильность данных
            stability_score = self._calculate_data_stability(aggregated_data[target_metric])
            
            # Выбираем модель на основе стабильности
            if stability_score > 0.7:
                # Стабильные данные - используем Random Forest
                model_result = self._run_random_forest_model(aggregated_data, target_metric, periods)
                model_result['model_used'] = 'Random Forest (стабильные данные)'
            else:
                # Нестабильные данные - используем Prophet
                if PROPHET_AVAILABLE:
                    model_result = self._run_prophet_model(aggregated_data, target_metric, periods)
                    model_result['model_used'] = 'Prophet (нестабильные данные)'
                else:
                    model_result = self._run_linear_model(aggregated_data, target_metric, periods)
                    model_result['model_used'] = 'Linear Regression (Prophet недоступен)'
            
            model_result['stability_score'] = stability_score
            return model_result
            
        except Exception as e:
            return {'success': False, 'message': f"Ошибка в гибридной модели: {str(e)}"}
    
    def _calculate_data_stability(self, data):
        """Расчет стабильности данных"""
        if len(data) < 3:
            return 0.5
        
        # Коэффициент вариации
        cv = data.std() / data.mean() if data.mean() != 0 else 1
        
        # Сила тренда
        x = np.arange(len(data))
        correlation = np.corrcoef(x, data)[0, 1]
        trend_strength = abs(correlation)
        
        # Количество выбросов
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        outlier_ratio = outliers / len(data)
        
        # Общий балл стабильности
        stability = (1 - min(cv, 1)) * 0.4 + trend_strength * 0.3 + (1 - outlier_ratio) * 0.3
        
        return max(0, min(1, stability))
    
    def _run_prophet_model(self, train_data, target_metric, periods):
        """Запуск модели Prophet"""
        try:
            if not PROPHET_AVAILABLE:
                return {'success': False, 'message': 'Prophet не установлен'}
            
            # Подготавливаем данные для Prophet
            prophet_data = train_data[['year', 'month', target_metric]].copy()
            prophet_data['ds'] = pd.to_datetime(prophet_data[['year', 'month']].assign(day=1))
            prophet_data = prophet_data.rename(columns={target_metric: 'y'})
            
            # Создаем и обучаем модель
            model = Prophet()
            model.fit(prophet_data[['ds', 'y']])
            
            # Создаем будущие периоды
            future_periods = []
            start_year, start_month = 2025, 9
            current_year, current_month = start_year, start_month
            
            for i in range(periods):
                future_periods.append({
                    'ds': pd.to_datetime(f'{current_year}-{current_month:02d}-01')
                })
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
            
            future_df = pd.DataFrame(future_periods)
            
            # Делаем прогноз
            forecast = model.predict(future_df)
            
            # Рассчитываем метрики
            y_true = prophet_data['y'].values
            y_pred = model.predict(prophet_data[['ds']])['yhat'].values
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            return {
                'success': True,
                'forecast_data': {
                    'periods': [
                        {
                            'date': row['ds'].strftime('%Y-%m'),
                            'forecast': max(0, row['yhat']),
                            'lower_bound': max(0, row['yhat_lower']),
                            'upper_bound': max(0, row['yhat_upper'])
                        }
                        for _, row in forecast.iterrows()
                    ],
                    'total_forecast': max(0, forecast['yhat'].sum()),
                    'growth_rate': ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100 if len(forecast) > 1 else 0
                },
                'accuracy': r2,
                'mae': mae,
                'chart_data': self._create_trend_chart_data(prophet_data, forecast)
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Ошибка в Prophet: {str(e)}"}
    
    def _run_random_forest_model(self, train_data, target_metric, periods):
        """Запуск модели Random Forest"""
        try:
            # Подготавливаем признаки
            features = self._prepare_time_features(train_data)
            target = train_data[target_metric].fillna(0)
            
            # Выбираем числовые признаки
            numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
            X = features[numeric_features].fillna(0)
            
            if len(X) == 0:
                return {'success': False, 'message': 'Нет подходящих признаков'}
            
            # Обучаем модель
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, target)
            
            # Рассчитываем метрики
            y_pred = model.predict(X)
            r2 = r2_score(target, y_pred)
            mae = mean_absolute_error(target, y_pred)
            
            # Создаем прогноз
            forecast_periods = self._create_future_periods(periods)
            future_features = self._prepare_time_features(forecast_periods)
            future_X = future_features[numeric_features].fillna(0)
            forecast_values = model.predict(future_X)
            forecast_values = np.maximum(forecast_values, 0)
            
            return {
                'success': True,
                'forecast_data': {
                    'periods': [
                        {
                            'date': f"{row['year']}-{row['month']:02d}",
                            'forecast': value,
                            'lower_bound': value * 0.8,
                            'upper_bound': value * 1.2
                        }
                        for (_, row), value in zip(forecast_periods.iterrows(), forecast_values)
                    ],
                    'total_forecast': forecast_values.sum(),
                    'growth_rate': ((forecast_values[-1] / forecast_values[0]) - 1) * 100 if len(forecast_values) > 1 else 0
                },
                'accuracy': r2,
                'mae': mae,
                'chart_data': self._create_trend_chart_data(train_data, forecast_periods, forecast_values)
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Ошибка в Random Forest: {str(e)}"}
    
    def _run_linear_model(self, train_data, target_metric, periods):
        """Запуск линейной модели"""
        try:
            # Подготавливаем признаки
            features = self._prepare_time_features(train_data)
            target = train_data[target_metric].fillna(0)
            
            # Выбираем числовые признаки
            numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
            X = features[numeric_features].fillna(0)
            
            if len(X) == 0:
                return {'success': False, 'message': 'Нет подходящих признаков'}
            
            # Обучаем модель
            model = LinearRegression()
            model.fit(X, target)
            
            # Рассчитываем метрики
            y_pred = model.predict(X)
            r2 = r2_score(target, y_pred)
            mae = mean_absolute_error(target, y_pred)
            
            # Создаем прогноз
            forecast_periods = self._create_future_periods(periods)
            future_features = self._prepare_time_features(forecast_periods)
            future_X = future_features[numeric_features].fillna(0)
            forecast_values = model.predict(future_X)
            forecast_values = np.maximum(forecast_values, 0)
            
            return {
                'success': True,
                'forecast_data': {
                    'periods': [
                        {
                            'date': f"{row['year']}-{row['month']:02d}",
                            'forecast': value,
                            'lower_bound': value * 0.9,
                            'upper_bound': value * 1.1
                        }
                        for (_, row), value in zip(forecast_periods.iterrows(), forecast_values)
                    ],
                    'total_forecast': forecast_values.sum(),
                    'growth_rate': ((forecast_values[-1] / forecast_values[0]) - 1) * 100 if len(forecast_values) > 1 else 0
                },
                'accuracy': r2,
                'mae': mae,
                'chart_data': self._create_trend_chart_data(train_data, forecast_periods, forecast_values)
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Ошибка в линейной модели: {str(e)}"}
    
    def _prepare_time_features(self, data):
        """Подготовка временных признаков"""
        features = data.copy()
        
        if 'year' in features.columns and 'month' in features.columns:
            features['time_index'] = (features['year'] - features['year'].min()) * 12 + (features['month'] - 1)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['quarter'] = ((features['month'] - 1) // 3) + 1
            
            # Квартальные признаки
            for q in range(1, 5):
                features[f'q{q}'] = (features['quarter'] == q).astype(int)
        
        return features
    
    def _create_future_periods(self, periods):
        """Создание будущих периодов"""
        start_year, start_month = 2025, 9
        
        periods_data = []
        current_year, current_month = start_year, start_month
        
        for i in range(periods):
            periods_data.append({
                'year': current_year,
                'month': current_month
            })
            
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return pd.DataFrame(periods_data)
    
    def _create_trend_chart_data(self, train_data, forecast_data, forecast_values=None):
        """Создание данных для графика тренда"""
        try:
            # Исторические данные
            historical_dates = [f"{row['year']}-{row['month']:02d}" for _, row in train_data.iterrows()]
            historical_values = train_data.iloc[:, -1].values if len(train_data.columns) > 2 else train_data.iloc[:, 0].values
            
            # Прогнозные данные
            if forecast_values is not None:
                forecast_dates = [f"{row['year']}-{row['month']:02d}" for _, row in forecast_data.iterrows()]
                forecast_vals = forecast_values
            else:
                # Для Prophet
                forecast_dates = [row['ds'].strftime('%Y-%m') for _, row in forecast_data.iterrows()]
                forecast_vals = forecast_data['yhat'].values
            
            return {
                'labels': historical_dates + forecast_dates,
                'datasets': [{
                    'label': 'Исторические данные',
                    'data': historical_values.tolist() + [None] * len(forecast_vals),
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.1
                }, {
                    'label': 'Прогноз',
                    'data': [None] * len(historical_values) + forecast_vals.tolist(),
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'tension': 0.1
                }]
            }
        except Exception as e:
            print(f"Ошибка создания графика: {str(e)}")
            return None

    # МОДУЛЬ 2: АНАЛИЗ ЗАВИСИМОСТЕЙ
    def run_dependency_analysis(self, config):
        """Запуск анализа зависимостей"""
        try:
            budget_field = config['budget_field']
            target_metrics = config['target_metrics']
            method = config['method']
            paid_channels = config['paid_channels']
            organic_channels = config['organic_channels']
            
            if budget_field not in self.df.columns:
                return False, f"Поле бюджета {budget_field} не найдено"
            
            dependencies = {}
            
            # Анализируем зависимости для каждой метрики
            for metric in target_metrics:
                if metric not in self.df.columns:
                    continue
                
                metric_dependencies = {}
                
                # Анализ для платных каналов
                if paid_channels and len(paid_channels) > 0:
                    paid_data = self.df[self.df['subdivision'].isin(paid_channels)]
                    if len(paid_data) > 10:
                        correlation = paid_data[budget_field].corr(paid_data[metric])
                        metric_dependencies['paid_channels'] = {
                            'correlation': correlation,
                            'data_points': len(paid_data),
                            'strength': self._get_correlation_strength(correlation)
                        }
                
                # Анализ для органических каналов
                if organic_channels and len(organic_channels) > 0:
                    organic_data = self.df[self.df['subdivision'].isin(organic_channels)]
                    if len(organic_data) > 10:
                        correlation = organic_data[budget_field].corr(organic_data[metric])
                        metric_dependencies['organic_channels'] = {
                            'correlation': correlation,
                            'data_points': len(organic_data),
                            'strength': self._get_correlation_strength(correlation)
                        }
                
                # Общий анализ
                if len(self.df) > 10:
                    correlation = self.df[budget_field].corr(self.df[metric])
                    metric_dependencies['overall'] = {
                        'correlation': correlation,
                        'data_points': len(self.df),
                        'strength': self._get_correlation_strength(correlation)
                    }
                
                if metric_dependencies:
                    dependencies[metric] = metric_dependencies
            
            self.dependency_results = {
                'dependencies': dependencies,
                'chart_data': self._create_dependency_chart_data(dependencies)
            }
            
            return True, f"Анализ зависимостей завершен для {len(dependencies)} метрик"
            
        except Exception as e:
            return False, f"Ошибка в анализе зависимостей: {str(e)}"
    
    def _get_correlation_strength(self, correlation):
        """Определение силы корреляции"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return 'Сильная'
        elif abs_corr >= 0.3:
            return 'Умеренная'
        else:
            return 'Слабая'
    
    def _create_dependency_chart_data(self, dependencies):
        """Создание данных для графика зависимостей"""
        try:
            labels = []
            correlations = []
            colors = []
            
            for metric, metric_deps in dependencies.items():
                for channel_type, dep_info in metric_deps.items():
                    labels.append(f"{metric}\n({channel_type})")
                    correlations.append(dep_info['correlation'])
                    
                    # Цвет в зависимости от силы корреляции
                    abs_corr = abs(dep_info['correlation'])
                    if abs_corr >= 0.7:
                        colors.append('rgba(255, 99, 132, 0.8)')  # Красный
                    elif abs_corr >= 0.3:
                        colors.append('rgba(255, 205, 86, 0.8)')  # Желтый
                    else:
                        colors.append('rgba(54, 162, 235, 0.8)')  # Синий
            
            return {
                'labels': labels,
                'datasets': [{
                    'label': 'Корреляция с бюджетом',
                    'data': correlations,
                    'backgroundColor': colors,
                    'borderColor': colors,
                    'borderWidth': 1
                }]
            }
        except Exception as e:
            print(f"Ошибка создания графика зависимостей: {str(e)}")
            return None

    # МОДУЛЬ 3: МОДЕЛИРОВАНИЕ
    def run_simulation(self, config):
        """Запуск моделирования сценариев"""
        try:
            simulation_field = config['simulation_field']
            periods = config['periods']
            scenarios = config['scenarios']
            
            if simulation_field not in self.df.columns:
                return False, f"Поле {simulation_field} не найдено"
            
            simulation_results = {}
            
            # Базовый прогноз
            base_forecast = self._create_base_forecast(simulation_field, periods)
            
            # Моделируем каждый сценарий
            for scenario in scenarios:
                scenario_name = scenario['name']
                change_percent = scenario['change']
                
                # Применяем изменение
                modified_forecast = base_forecast * (1 + change_percent / 100)
                
                # Рассчитываем влияние на другие метрики
                impact_analysis = self._analyze_scenario_impact(
                    simulation_field, change_percent, modified_forecast
                )
                
                simulation_results[scenario_name] = {
                    'scenario': scenario,
                    'modified_forecast': modified_forecast.tolist(),
                    'impact_analysis': impact_analysis
                }
            
            self.simulation_results = {
                'scenarios': simulation_results,
                'chart_data': self._create_simulation_chart_data(simulation_results)
            }
            
            return True, f"Моделирование завершено для {len(scenarios)} сценариев"
            
        except Exception as e:
            return False, f"Ошибка в моделировании: {str(e)}"
    
    def _create_base_forecast(self, field, periods):
        """Создание базового прогноза"""
        try:
            # Используем среднее значение за последние периоды
            recent_data = self.df[field].dropna().tail(12)  # Последние 12 месяцев
            if len(recent_data) == 0:
                recent_data = self.df[field].dropna()
            
            if len(recent_data) == 0:
                return np.zeros(periods)
            
            base_value = recent_data.mean()
            return np.full(periods, base_value)
            
        except Exception as e:
            print(f"Ошибка создания базового прогноза: {str(e)}")
            return np.zeros(periods)
    
    def _analyze_scenario_impact(self, simulation_field, change_percent, modified_forecast):
        """Анализ влияния сценария на другие метрики"""
        try:
            impact_analysis = {}
            
            # Простой анализ влияния на выручку
            if 'revenue_total' in self.df.columns and simulation_field == 'ads_cost':
                # Предполагаем, что увеличение рекламного бюджета на 1% дает 0.5% роста выручки
                revenue_impact = change_percent * 0.5
                impact_analysis['revenue_impact'] = revenue_impact
            
            # Анализ влияния на трафик
            if 'traffic_total' in self.df.columns and simulation_field == 'ads_cost':
                # Предполагаем, что увеличение рекламного бюджета на 1% дает 0.8% роста трафика
                traffic_impact = change_percent * 0.8
                impact_analysis['traffic_impact'] = traffic_impact
            
            return impact_analysis
            
        except Exception as e:
            print(f"Ошибка анализа влияния: {str(e)}")
            return {}
    
    def _create_simulation_chart_data(self, simulation_results):
        """Создание данных для графика моделирования"""
        try:
            labels = []
            revenue_impacts = []
            traffic_impacts = []
            
            for scenario_name, result in simulation_results.items():
                labels.append(scenario_name)
                impact = result['impact_analysis']
                revenue_impacts.append(impact.get('revenue_impact', 0))
                traffic_impacts.append(impact.get('traffic_impact', 0))
            
            return {
                'labels': labels,
                'datasets': [{
                    'label': 'Влияние на выручку (%)',
                    'data': revenue_impacts,
                    'backgroundColor': 'rgba(75, 192, 192, 0.8)',
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'borderWidth': 1
                }, {
                    'label': 'Влияние на трафик (%)',
                    'data': traffic_impacts,
                    'backgroundColor': 'rgba(255, 99, 132, 0.8)',
                    'borderColor': 'rgba(255, 99, 132, 1)',
                    'borderWidth': 1
                }]
            }
        except Exception as e:
            print(f"Ошибка создания графика моделирования: {str(e)}")
            return None

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    def save_results(self, session_id):
        """Сохранение результатов"""
        try:
            all_results = []
            
            # Добавляем результаты трендового прогноза
            if self.trend_results and 'forecast_data' in self.trend_results:
                for period in self.trend_results['forecast_data']['periods']:
                    all_results.append({
                        'module': 'trend_forecast',
                        'date': period['date'],
                        'forecast': period['forecast'],
                        'lower_bound': period['lower_bound'],
                        'upper_bound': period['upper_bound']
                    })
            
            # Добавляем результаты анализа зависимостей
            if self.dependency_results and 'dependencies' in self.dependency_results:
                for metric, metric_deps in self.dependency_results['dependencies'].items():
                    for channel_type, dep_info in metric_deps.items():
                        all_results.append({
                            'module': 'dependency_analysis',
                            'metric': metric,
                            'channel_type': channel_type,
                            'correlation': dep_info['correlation'],
                            'strength': dep_info['strength']
                        })
            
            # Добавляем результаты моделирования
            if self.simulation_results and 'scenarios' in self.simulation_results:
                for scenario_name, result in self.simulation_results['scenarios'].items():
                    all_results.append({
                        'module': 'simulation',
                        'scenario': scenario_name,
                        'revenue_impact': result['impact_analysis'].get('revenue_impact', 0),
                        'traffic_impact': result['impact_analysis'].get('traffic_impact', 0)
                    })
            
            # Создаем DataFrame и сохраняем
            if all_results:
                results_df = pd.DataFrame(all_results)
                filename = f"marfor_results_{session_id}.csv"
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                results_df.to_csv(filepath, index=False, encoding='utf-8')
                return filepath
            
            return None
            
        except Exception as e:
            print(f"Ошибка сохранения результатов: {str(e)}")
            return None

# Глобальный объект приложения
marfor_app = MARFORWebApp()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('marfor_interface.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файла"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Файл не выбран'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Файл не выбран'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        filename = f"{session_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Загружаем данные
        success, message = marfor_app.load_data_from_file(filepath)
        
        if success:
            marfor_app.session_id = session_id
            data_info = marfor_app.get_data_info()
            return jsonify({
                'success': True, 
                'message': message,
                'session_id': session_id,
                'data_info': data_info
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Недопустимый тип файла'})

@app.route('/trend_forecast', methods=['POST'])
def trend_forecast():
    """Трендовый прогноз"""
    config = request.json
    
    success, message = marfor_app.run_trend_forecast(config)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'forecast_data': marfor_app.trend_results['forecast_data'],
            'model_used': marfor_app.trend_results.get('model_used', 'Unknown'),
            'accuracy': marfor_app.trend_results.get('accuracy', 0),
            'chart_data': marfor_app.trend_results.get('chart_data')
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/dependency_analysis', methods=['POST'])
def dependency_analysis():
    """Анализ зависимостей"""
    config = request.json
    
    success, message = marfor_app.run_dependency_analysis(config)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'dependencies': marfor_app.dependency_results['dependencies'],
            'chart_data': marfor_app.dependency_results.get('chart_data')
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/simulation', methods=['POST'])
def simulation():
    """Моделирование сценариев"""
    config = request.json
    
    success, message = marfor_app.run_simulation(config)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'scenarios': marfor_app.simulation_results['scenarios'],
            'chart_data': marfor_app.simulation_results.get('chart_data')
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/download/csv/<session_id>')
def download_csv(session_id):
    """Скачивание результатов в CSV"""
    filename = f"marfor_results_{session_id}.csv"
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        return jsonify({'success': False, 'message': 'Файл не найден'})

@app.route('/download/excel/<session_id>')
def download_excel(session_id):
    """Скачивание результатов в Excel"""
    filename = f"marfor_results_{session_id}.xlsx"
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    try:
        # Создаем Excel файл
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Лист с трендовым прогнозом
            if marfor_app.trend_results and 'forecast_data' in marfor_app.trend_results:
                trend_df = pd.DataFrame(marfor_app.trend_results['forecast_data']['periods'])
                trend_df.to_excel(writer, sheet_name='Трендовый прогноз', index=False)
            
            # Лист с анализом зависимостей
            if marfor_app.dependency_results and 'dependencies' in marfor_app.dependency_results:
                dep_data = []
                for metric, metric_deps in marfor_app.dependency_results['dependencies'].items():
                    for channel_type, dep_info in metric_deps.items():
                        dep_data.append({
                            'Метрика': metric,
                            'Тип канала': channel_type,
                            'Корреляция': dep_info['correlation'],
                            'Сила': dep_info['strength']
                        })
                if dep_data:
                    dep_df = pd.DataFrame(dep_data)
                    dep_df.to_excel(writer, sheet_name='Анализ зависимостей', index=False)
            
            # Лист с моделированием
            if marfor_app.simulation_results and 'scenarios' in marfor_app.simulation_results:
                sim_data = []
                for scenario_name, result in marfor_app.simulation_results['scenarios'].items():
                    sim_data.append({
                        'Сценарий': scenario_name,
                        'Влияние на выручку (%)': result['impact_analysis'].get('revenue_impact', 0),
                        'Влияние на трафик (%)': result['impact_analysis'].get('traffic_impact', 0)
                    })
                if sim_data:
                    sim_df = pd.DataFrame(sim_data)
                    sim_df.to_excel(writer, sheet_name='Моделирование', index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка создания Excel файла: {str(e)}'})

@app.route('/download/report/<session_id>')
def download_report(session_id):
    """Скачивание отчета в PDF (заглушка)"""
    return jsonify({'success': False, 'message': 'Функция создания PDF отчета в разработке'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
