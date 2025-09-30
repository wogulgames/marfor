#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARFOR - Каскадный трендовый прогноз
Веб-приложение для настройки маппинга данных и создания каскадного прогноза
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
app.secret_key = 'marfor-cascaded-forecast-2024'

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

class CascadedForecastApp:
    def __init__(self):
        """Инициализация приложения каскадного прогноза"""
        self.df = None
        self.session_id = None
        self.data_mapping = {}
        self.forecast_results = {}
        
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
        
        # Конвертируем numpy типы в стандартные Python типы для JSON сериализации
        for col in self.df.columns:
            if self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype('int32')
            elif self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')
        
        print(f"Очистка данных: удалено {initial_count - len(self.df)} записей")
    
    def apply_data_transformations(self, field_type_changes):
        """Применение изменений типов полей"""
        try:
            if self.df is None:
                return False, "Данные не загружены"
            
            for change in field_type_changes:
                field = change['field']
                new_type = change['new_type']
                fill_method = change.get('fill_method', 'zero')
                
                if field not in self.df.columns:
                    continue
                
                if new_type == 'numeric':
                    # Преобразуем в числовые данные
                    if fill_method == 'zero':
                        self.df[field] = self.df[field].fillna(0)
                    elif fill_method == 'mean':
                        mean_val = pd.to_numeric(self.df[field], errors='coerce').mean()
                        self.df[field] = self.df[field].fillna(mean_val if not pd.isna(mean_val) else 0)
                    elif fill_method == 'median':
                        median_val = pd.to_numeric(self.df[field], errors='coerce').median()
                        self.df[field] = self.df[field].fillna(median_val if not pd.isna(median_val) else 0)
                    
                    # Преобразуем в числовой тип
                    self.df[field] = pd.to_numeric(self.df[field], errors='coerce').fillna(0)
                    
                elif new_type == 'categorical':
                    # Преобразуем в категориальные данные
                    self.df[field] = self.df[field].astype('category')
                    
                elif new_type == 'date':
                    # Парсинг дат
                    self.df[field] = pd.to_datetime(self.df[field], errors='coerce')
                    
                elif new_type == 'text':
                    # Преобразуем в текстовые данные
                    self.df[field] = self.df[field].astype(str)
            
            # Конвертируем numpy типы в стандартные Python типы для JSON сериализации
            for col in self.df.columns:
                if self.df[col].dtype == 'int64':
                    self.df[col] = self.df[col].astype('int32')
                elif self.df[col].dtype == 'float64':
                    self.df[col] = self.df[col].astype('float32')
            
            return True, "Преобразования данных успешно применены"
            
        except Exception as e:
            return False, f"Ошибка при применении преобразований: {str(e)}"
    
    def _convert_to_json_safe(self, obj):
        """Конвертация объекта в JSON-безопасный формат"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_data_info(self):
        """Получение информации о данных для маппинга"""
        if self.df is None:
            return None
        
        # Анализируем типы данных
        column_info = {}
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            
            # Определяем тип данных
            if len(col_data) == 0:
                data_type = 'empty'
            elif col_data.dtype in ['int64', 'int32', 'float64', 'float32']:
                data_type = 'numeric'
            else:
                # Проверяем, можно ли преобразовать в дату
                try:
                    pd.to_datetime(col_data.head(10), errors='raise')
                    data_type = 'date'
                except:
                    # Проверяем уникальные значения для определения типа
                    unique_count = int(col_data.nunique())
                    total_count = int(len(col_data))
                    
                    if unique_count <= 12 and total_count > 0:
                        # Возможно, это категориальный признак
                        data_type = 'categorical'
                    elif unique_count / total_count > 0.8:
                        # Много уникальных значений - возможно, текст
                        data_type = 'text'
                    else:
                        data_type = 'categorical'
            
            # Получаем примеры значений и конвертируем в JSON-совместимые типы
            sample_values = []
            for val in col_data.head(5):
                sample_values.append(self._convert_to_json_safe(val))
            
            column_info[col] = {
                'type': data_type,
                'sample_values': sample_values,
                'unique_count': self._convert_to_json_safe(col_data.nunique()),
                'null_count': self._convert_to_json_safe(self.df[col].isnull().sum()),
                'dtype': str(self.df[col].dtype)
            }
        
        # Конвертируем sample_data в JSON-совместимый формат
        sample_data = []
        for _, row in self.df.head(5).iterrows():
            sample_row = {}
            for col, val in row.items():
                sample_row[col] = self._convert_to_json_safe(val)
            sample_data.append(sample_row)
        
        info = {
            'shape': [self._convert_to_json_safe(self.df.shape[0]), self._convert_to_json_safe(self.df.shape[1])],
            'columns': [str(col) for col in self.df.columns],
            'column_info': column_info,
            'sample_data': sample_data
        }
        
        return info
    
    def set_data_mapping(self, mapping_config):
        """Установка маппинга данных"""
        self.data_mapping = mapping_config
        
        # Отладочная информация
        print(f"Получен маппинг: {mapping_config}")
        print(f"Метрики: {mapping_config.get('metrics', [])}")
        
        # Валидируем маппинг
        validation_result = self._validate_mapping()
        
        if validation_result['valid']:
            return True, "Маппинг данных успешно настроен"
        else:
            return False, f"Ошибка в маппинге: {validation_result['error']}"
    
    def _validate_mapping(self):
        """Валидация маппинга данных"""
        try:
            # Проверяем временные поля
            time_fields = self.data_mapping.get('time_fields', {})
            if not time_fields:
                return {'valid': False, 'error': 'Не заданы временные поля'}
            
            for time_type, field in time_fields.items():
                if field and field not in self.df.columns:
                    return {'valid': False, 'error': f'Поле {field} не найдено в данных'}
            
            # Проверяем признаки
            dimensions = self.data_mapping.get('dimensions', [])
            for dim in dimensions:
                if dim['field'] not in self.df.columns:
                    return {'valid': False, 'error': f'Поле признака {dim["field"]} не найдено в данных'}
            
            # Проверяем метрики
            metrics = self.data_mapping.get('metrics', [])
            if not metrics:
                return {'valid': False, 'error': 'Не заданы метрики для прогноза'}
            
            for metric in metrics:
                if metric['field'] not in self.df.columns:
                    return {'valid': False, 'error': f'Поле метрики {metric["field"]} не найдено в данных'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Ошибка валидации: {str(e)}'}
    
    def run_cascaded_forecast(self, forecast_config):
        """Запуск каскадного прогноза"""
        try:
            # Подготавливаем данные
            prepared_data = self._prepare_data_for_forecast()
            
            if prepared_data is None:
                return False, "Ошибка подготовки данных"
            
            # Создаем каскадную структуру
            cascade_structure = self._create_cascade_structure(prepared_data)
            
            # Выполняем прогноз на каждом уровне каскада
            forecast_results = {}
            
            for level, level_data in cascade_structure.items():
                level_forecast = self._forecast_level(level_data, forecast_config)
                if level_forecast:
                    forecast_results[level] = level_forecast
            
            # Оцениваем точность на исторических данных
            accuracy_results = self._evaluate_accuracy(prepared_data)
            
            self.forecast_results = {
                'forecast_results': forecast_results,
                'accuracy_results': accuracy_results,
                'cascade_structure': cascade_structure
            }
            
            return True, "Каскадный прогноз успешно выполнен"
            
        except Exception as e:
            return False, f"Ошибка в каскадном прогнозе: {str(e)}"
    
    def _prepare_data_for_forecast(self):
        """Подготовка данных для прогноза"""
        try:
            # Создаем временной индекс
            time_fields = self.data_mapping.get('time_fields', {})
            print(f"Временные поля: {time_fields}")
            
            if 'year' in time_fields and time_fields['year']:
                year_field = time_fields['year']
                print(f"Поле года: {year_field}")
            else:
                print("Ошибка: не найдено поле года")
                return None
            
            # Создаем временной индекс
            if 'month' in time_fields and time_fields['month']:
                month_field = time_fields['month']
                self.df['time_index'] = (self.df[year_field] - self.df[year_field].min()) * 12 + (self.df[month_field] - 1)
            elif 'quarter' in time_fields and time_fields['quarter']:
                quarter_field = time_fields['quarter']
                self.df['time_index'] = (self.df[year_field] - self.df[year_field].min()) * 4 + (self.df[quarter_field] - 1)
            else:
                self.df['time_index'] = self.df[year_field] - self.df[year_field].min()
            
            # Добавляем временные признаки
            self.df['year'] = self.df[year_field]
            if 'month' in time_fields and time_fields['month']:
                self.df['month'] = self.df[month_field]
            if 'quarter' in time_fields and time_fields['quarter']:
                self.df['quarter'] = self.df[quarter_field]
            
            return self.df.copy()
            
        except Exception as e:
            print(f"Ошибка подготовки данных: {str(e)}")
            return None
    
    def _create_cascade_structure(self, data):
        """Создание каскадной структуры данных"""
        dimensions = self.data_mapping.get('dimensions', [])
        metrics = self.data_mapping.get('metrics', [])
        
        cascade_structure = {}
        
        # Уровень 0: агрегированные данные
        level_0_data = data.groupby(['year', 'month']).agg({
            metric['field']: 'sum' for metric in metrics
        }).reset_index()
        cascade_structure['level_0_aggregated'] = level_0_data
        
        # Создаем уровни каскада
        for i, dimension in enumerate(dimensions):
            level_name = f'level_{i+1}_{dimension["name"]}'
            
            # Агрегируем данные по текущему измерению
            groupby_cols = ['year', 'month', dimension['field']]
            level_data = data.groupby(groupby_cols).agg({
                metric['field']: 'sum' for metric in metrics
            }).reset_index()
            
            cascade_structure[level_name] = level_data
        
        return cascade_structure
    
    def _forecast_level(self, level_data, forecast_config):
        """Прогноз на конкретном уровне каскада"""
        try:
            metrics = self.data_mapping.get('metrics', [])
            forecast_results = {}
            
            for metric in metrics:
                metric_field = metric['field']
                
                if metric_field not in level_data.columns:
                    continue
                
                # Подготавливаем данные для модели
                model_data = level_data[['year', 'month', metric_field]].copy()
                model_data = model_data.dropna()
                
                if len(model_data) < 10:
                    continue
                
                # Создаем временные признаки
                model_data['time_index'] = (model_data['year'] - model_data['year'].min()) * 12 + (model_data['month'] - 1)
                model_data['month_sin'] = np.sin(2 * np.pi * model_data['month'] / 12)
                model_data['month_cos'] = np.cos(2 * np.pi * model_data['month'] / 12)
                
                # Выбираем модель
                model_type = forecast_config.get('model_type', 'hybrid')
                
                if model_type == 'prophet' and PROPHET_AVAILABLE:
                    forecast_result = self._prophet_forecast(model_data, metric_field, forecast_config)
                elif model_type == 'random_forest':
                    forecast_result = self._random_forest_forecast(model_data, metric_field, forecast_config)
                else:
                    forecast_result = self._linear_forecast(model_data, metric_field, forecast_config)
                
                if forecast_result:
                    forecast_results[metric_field] = forecast_result
            
            return forecast_results
            
        except Exception as e:
            print(f"Ошибка прогноза уровня: {str(e)}")
            return None
    
    def _prophet_forecast(self, data, metric_field, config):
        """Прогноз с помощью Prophet"""
        try:
            if not PROPHET_AVAILABLE:
                return None
            
            # Подготавливаем данные для Prophet
            prophet_data = data[['year', 'month', metric_field]].copy()
            prophet_data['ds'] = pd.to_datetime(prophet_data[['year', 'month']].assign(day=1))
            prophet_data = prophet_data.rename(columns={metric_field: 'y'})
            
            # Создаем и обучаем модель
            model = Prophet()
            model.fit(prophet_data[['ds', 'y']])
            
            # Создаем будущие периоды
            periods = config.get('periods', 12)
            future_periods = self._create_future_periods(periods)
            future_df = pd.DataFrame(future_periods)
            
            # Делаем прогноз
            forecast = model.predict(future_df)
            
            return {
                'model_type': 'prophet',
                'forecast_periods': [
                    {
                        'year': int(row['ds'].year),
                        'month': int(row['ds'].month),
                        'forecast': float(max(0, row['yhat'])),
                        'lower_bound': float(max(0, row['yhat_lower'])),
                        'upper_bound': float(max(0, row['yhat_upper']))
                    }
                    for _, row in forecast.iterrows()
                ],
                'total_forecast': float(max(0, forecast['yhat'].sum()))
            }
            
        except Exception as e:
            print(f"Ошибка Prophet: {str(e)}")
            return None
    
    def _random_forest_forecast(self, data, metric_field, config):
        """Прогноз с помощью Random Forest"""
        try:
            # Подготавливаем признаки
            features = ['time_index', 'month_sin', 'month_cos']
            X = data[features].fillna(0)
            y = data[metric_field].fillna(0)
            
            # Обучаем модель
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Создаем прогноз
            periods = config.get('periods', 12)
            future_periods = self._create_future_periods(periods)
            
            future_features = []
            for period in future_periods:
                time_index = (period['year'] - data['year'].min()) * 12 + (period['month'] - 1)
                month_sin = np.sin(2 * np.pi * period['month'] / 12)
                month_cos = np.cos(2 * np.pi * period['month'] / 12)
                future_features.append([time_index, month_sin, month_cos])
            
            future_X = np.array(future_features)
            forecast_values = model.predict(future_X)
            forecast_values = np.maximum(forecast_values, 0)
            
            return {
                'model_type': 'random_forest',
                'forecast_periods': [
                    {
                        'year': int(period['year']),
                        'month': int(period['month']),
                        'forecast': float(value),
                        'lower_bound': float(value * 0.8),
                        'upper_bound': float(value * 1.2)
                    }
                    for period, value in zip(future_periods, forecast_values)
                ],
                'total_forecast': float(forecast_values.sum())
            }
            
        except Exception as e:
            print(f"Ошибка Random Forest: {str(e)}")
            return None
    
    def _linear_forecast(self, data, metric_field, config):
        """Прогноз с помощью линейной регрессии"""
        try:
            # Подготавливаем признаки
            features = ['time_index', 'month_sin', 'month_cos']
            X = data[features].fillna(0)
            y = data[metric_field].fillna(0)
            
            # Обучаем модель
            model = LinearRegression()
            model.fit(X, y)
            
            # Создаем прогноз
            periods = config.get('periods', 12)
            future_periods = self._create_future_periods(periods)
            
            future_features = []
            for period in future_periods:
                time_index = (period['year'] - data['year'].min()) * 12 + (period['month'] - 1)
                month_sin = np.sin(2 * np.pi * period['month'] / 12)
                month_cos = np.cos(2 * np.pi * period['month'] / 12)
                future_features.append([time_index, month_sin, month_cos])
            
            future_X = np.array(future_features)
            forecast_values = model.predict(future_X)
            forecast_values = np.maximum(forecast_values, 0)
            
            return {
                'model_type': 'linear',
                'forecast_periods': [
                    {
                        'year': int(period['year']),
                        'month': int(period['month']),
                        'forecast': float(value),
                        'lower_bound': float(value * 0.9),
                        'upper_bound': float(value * 1.1)
                    }
                    for period, value in zip(future_periods, forecast_values)
                ],
                'total_forecast': float(forecast_values.sum())
            }
            
        except Exception as e:
            print(f"Ошибка линейной регрессии: {str(e)}")
            return None
    
    def _create_future_periods(self, periods):
        """Создание будущих периодов"""
        # Определяем последний период в данных
        last_year = self.df['year'].max()
        last_month = self.df[self.df['year'] == last_year]['month'].max() if 'month' in self.df.columns else 1
        
        # Начинаем с следующего месяца
        start_year = last_year
        start_month = last_month + 1 if last_month < 12 else 1
        if last_month == 12:
            start_year += 1
        
        future_periods = []
        current_year, current_month = start_year, start_month
        
        for i in range(periods):
            future_periods.append({
                'year': current_year,
                'month': current_month
            })
            
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return future_periods
    
    def _evaluate_accuracy(self, data):
        """Оценка точности на исторических данных"""
        try:
            metrics = self.data_mapping.get('metrics', [])
            accuracy_results = {}
            
            for metric in metrics:
                metric_field = metric['field']
                
                if metric_field not in data.columns:
                    continue
                
                # Разделяем данные на обучающую и тестовую выборки
                train_data = data[data['year'] < data['year'].max()].copy()
                test_data = data[data['year'] == data['year'].max()].copy()
                
                if len(train_data) < 10 or len(test_data) < 3:
                    continue
                
                # Обучаем модель на обучающих данных
                train_features = ['time_index', 'month_sin', 'month_cos']
                X_train = train_data[train_features].fillna(0)
                y_train = train_data[metric_field].fillna(0)
                
                X_test = test_data[train_features].fillna(0)
                y_test = test_data[metric_field].fillna(0)
                
                # Тестируем разные модели
                models = {
                    'linear': LinearRegression(),
                    'ridge': Ridge(alpha=1.0),
                    'lasso': Lasso(alpha=1.0),
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
                }
                
                model_results = {}
                for model_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        model_results[model_name] = {
                            'r2': float(r2),
                            'mae': float(mae),
                            'rmse': float(rmse)
                        }
                    except:
                        continue
                
                if model_results:
                    # Выбираем лучшую модель
                    best_model = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
                    accuracy_results[metric_field] = {
                        'best_model': best_model,
                        'model_results': model_results
                    }
            
            return accuracy_results
            
        except Exception as e:
            print(f"Ошибка оценки точности: {str(e)}")
            return {}
    
    def save_results(self, session_id):
        """Сохранение результатов"""
        try:
            all_results = []
            
            if self.forecast_results and 'forecast_results' in self.forecast_results:
                for level, level_results in self.forecast_results['forecast_results'].items():
                    for metric, metric_results in level_results.items():
                        for period in metric_results['forecast_periods']:
                            all_results.append({
                                'level': level,
                                'metric': metric,
                                'year': int(period['year']),
                                'month': int(period['month']),
                                'forecast': float(period['forecast']),
                                'lower_bound': float(period['lower_bound']),
                                'upper_bound': float(period['upper_bound'])
                            })
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                filename = f"cascaded_forecast_{session_id}.csv"
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                results_df.to_csv(filepath, index=False, encoding='utf-8')
                return filepath
            
            return None
            
        except Exception as e:
            print(f"Ошибка сохранения результатов: {str(e)}")
            return None

# Глобальный объект приложения
cascaded_app = CascadedForecastApp()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('cascaded_forecast.html')

@app.route('/favicon.ico')
def favicon():
    """Обработка favicon"""
    return '', 204  # No Content

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
        success, message = cascaded_app.load_data_from_file(filepath)
        
        if success:
            cascaded_app.session_id = session_id
            data_info = cascaded_app.get_data_info()
            return jsonify({
                'success': True, 
                'message': message,
                'session_id': session_id,
                'data_info': data_info
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Недопустимый тип файла'})

@app.route('/apply_transformations', methods=['POST'])
def apply_transformations():
    """Применение изменений типов полей"""
    field_type_changes = request.json.get('field_changes', [])
    
    success, message = cascaded_app.apply_data_transformations(field_type_changes)
    
    if success:
        # Обновляем информацию о данных после преобразований
        data_info = cascaded_app.get_data_info()
        return jsonify({
            'success': True,
            'message': message,
            'data_info': data_info
        })
    else:
        return jsonify({
            'success': False,
            'message': message
        })

@app.route('/set_mapping', methods=['POST'])
def set_mapping():
    """Установка маппинга данных"""
    mapping_config = request.json
    
    success, message = cascaded_app.set_data_mapping(mapping_config)
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/run_forecast', methods=['POST'])
def run_forecast():
    """Запуск каскадного прогноза"""
    forecast_config = request.json
    
    success, message = cascaded_app.run_cascaded_forecast(forecast_config)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'forecast_results': cascaded_app.forecast_results['forecast_results'],
            'accuracy_results': cascaded_app.forecast_results['accuracy_results']
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/download/<session_id>')
def download_results(session_id):
    """Скачивание результатов"""
    filename = f"cascaded_forecast_{session_id}.csv"
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        return jsonify({'success': False, 'message': 'Файл не найден'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
