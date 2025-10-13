#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARFOR - Рабочее веб-приложение для прогнозирования маркетинговых данных
Интегрирует каскадную модель с Random Forest и веб-интерфейс
"""

import pandas as pd
import numpy as np
import os
import uuid
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def convert_to_json_serializable(obj):
    """Конвертация pandas/numpy объектов в JSON-совместимые типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# Flask
from flask import Flask, render_template, render_template_string, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename

# Scikit-learn модели
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__, static_folder='static')
app.secret_key = 'marfor-working-app-2024'

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

class WorkingForecastApp:
    def __init__(self):
        """Инициализация рабочего приложения прогнозирования"""
        self.df = None
        self.session_id = None
        self.forecast_results = {}
        self.data_mapping = {}
        
    def load_data_from_file(self, file_path: str):
        """Загрузка данных из файла"""
        try:
            if file_path.endswith('.csv'):
                # Пробуем разные разделители
                separators = [',', ';', '\t', '|']
                for sep in separators:
                    try:
                        self.df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                        if len(self.df.columns) > 1:
                            print(f"✅ Загружено с разделителем '{sep}': {len(self.df)} записей, {len(self.df.columns)} колонок")
                            break
                    except:
                        try:
                            self.df = pd.read_csv(file_path, sep=sep, encoding='cp1251')
                            if len(self.df.columns) > 1:
                                print(f"✅ Загружено с разделителем '{sep}' (cp1251): {len(self.df)} записей, {len(self.df.columns)} колонок")
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
        """Очистка и подготовка данных"""
        print(f"\n🧹 ОЧИСТКА ДАННЫХ:")
        initial_count = len(self.df)
        
        # Очистка числовых колонок
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Пробуем преобразовать в числовой формат
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        
        # Удаляем строки где все значения NaN
        self.df = self.df.dropna(how='all')
        
        print(f"  Удалено {initial_count - len(self.df)} записей с пустыми данными")
        print(f"  После очистки: {len(self.df)} записей")
        
        # Анализ колонок
        print(f"\n📊 АНАЛИЗ КОЛОНОК:")
        for i, col in enumerate(self.df.columns):
            dtype = self.df[col].dtype
            non_null = self.df[col].count()
            print(f"  {i}: {col} ({dtype}) - {non_null} значений")
    
    def get_data_info(self):
        """Получение информации о данных"""
        if self.df is None:
            return None
        
        # Очищаем NaN значения для JSON
        sample_data = convert_to_json_serializable(self.df.head(5).fillna('').to_dict('records'))
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'sample_data': sample_data,
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return info
    
    def apply_data_mapping(self, mapping_config):
        """Применить маппинг данных"""
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        df = self.df.copy()
        
        # Обработка колонок
        columns_to_include = []
        for col_config in mapping_config.get('columns', []):
            if col_config.get('include', True):
                col_name = col_config['name']
                col_type = col_config.get('type', 'auto')
                
                if col_name in df.columns:
                    # Преобразование типов
                    if col_type == 'numeric':
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    elif col_type == 'text':
                        df[col_name] = df[col_name].astype(str)
                    elif col_type == 'category':
                        df[col_name] = df[col_name].astype('category')
                    
                    columns_to_include.append(col_name)
        
        # Оставляем только выбранные колонки
        if columns_to_include:
            df = df[columns_to_include]
        
        # Обработка пустых значений
        missing_strategy = mapping_config.get('missingValues', 'zeros')
        if missing_strategy == 'remove':
            df = df.dropna()
        elif missing_strategy == 'zeros':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Обработка выбросов
        if mapping_config.get('detectOutliers', False):
            threshold = mapping_config.get('outlierThreshold', 3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
                if mapping_config.get('removeOutliers', False):
                    df = df[~outliers]
                else:
                    # Заменяем выбросы на медиану
                    df.loc[outliers, col] = df[col].median()
        
        # Нормализация данных
        if mapping_config.get('normalizeData', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Логарифмическое преобразование
        if mapping_config.get('logTransform', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
        
        # Создание временных признаков
        if mapping_config.get('createFeatures', False):
            time_series = mapping_config.get('timeSeries', {})
            
            # Обработка временных рядов
            if time_series.get('date'):
                date_col = df.columns[int(time_series['date'])]
                if date_col in df.columns:
                    df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                    df['year'] = df['date_parsed'].dt.year
                    df['month'] = df['date_parsed'].dt.month
                    df['quarter'] = df['date_parsed'].dt.quarter
                    df['week'] = df['date_parsed'].dt.isocalendar().week
            
            if time_series.get('year'):
                year_col = df.columns[int(time_series['year'])]
                if year_col in df.columns:
                    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
            
            if time_series.get('month'):
                month_col = df.columns[int(time_series['month'])]
                if month_col in df.columns:
                    # Обработка текстовых месяцев
                    month_mapping = {
                    'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4,
                    'май': 5, 'июнь': 6, 'июль': 7, 'август': 8,
                    'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
                    }
                    if df[month_col].dtype == 'object':
                        df['month'] = df[month_col].str.lower().map(month_mapping).fillna(pd.to_numeric(df[month_col], errors='coerce'))
                else:
                    df['month'] = pd.to_numeric(df[month_col], errors='coerce')
            
            if time_series.get('quarter'):
                quarter_col = df.columns[int(time_series['quarter'])]
                if quarter_col in df.columns:
                    df['quarter'] = pd.to_numeric(df[quarter_col], errors='coerce')
            
            if time_series.get('week'):
                week_col = df.columns[int(time_series['week'])]
                if week_col in df.columns:
                    df['week'] = pd.to_numeric(df[week_col], errors='coerce')
            
            if time_series.get('halfyear'):
                halfyear_col = df.columns[int(time_series['halfyear'])]
                if halfyear_col in df.columns:
                    df['halfyear'] = pd.to_numeric(df[halfyear_col], errors='coerce')
            
            # Создаем дополнительные временные признаки
            if 'year' in df.columns and 'month' in df.columns:
                df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Зима
                                       3: 1, 4: 1, 5: 1,    # Весна
                                       6: 2, 7: 2, 8: 2,    # Лето
                                       9: 3, 10: 3, 11: 3}) # Осень
                df['is_weekend'] = 0  # Заглушка для будущего расширения
        
        # Обновляем данные в экземпляре
        self.df = df
        
        return df
    
    def set_data_mapping(self, mapping):
        """Установка маппинга колонок"""
        self.data_mapping = mapping
        print(f"\n🗺️ МАППИНГ ДАННЫХ:")
        for key, value in mapping.items():
            print(f"  {key}: колонка {value}")
    
    def run_cascaded_forecast(self, config):
        """Запуск каскадного прогноза с Random Forest"""
        try:
            print(f"\n🔮 ЗАПУСК КАСКАДНОГО ПРОГНОЗА:")
            
            # Получаем настройки
            periods = config.get('periods', 4)
            method = config.get('method', 'random_forest')
            year_col = self.data_mapping.get('year', 0)
            month_col = self.data_mapping.get('month', 1)
            
            print(f"  Периодов прогноза: {periods}")
            print(f"  Метод: {method}")
            print(f"  Колонка года: {year_col}")
            print(f"  Колонка месяца: {month_col}")
            
            # Проверяем наличие временных колонок
            if year_col >= len(self.df.columns) or month_col >= len(self.df.columns):
                return False, "Неправильно указаны колонки года или месяца"
            
            # Переименовываем колонки для удобства
            year_col_name = self.df.columns[year_col]
            month_col_name = self.df.columns[month_col]
            
            # Очищаем временные данные
            self.df[year_col_name] = pd.to_numeric(self.df[year_col_name], errors='coerce')
            self.df[month_col_name] = pd.to_numeric(self.df[month_col_name], errors='coerce')
            
            # Удаляем строки с пустыми временными данными
            self.df = self.df.dropna(subset=[year_col_name, month_col_name])
            
            if len(self.df) < 10:
                return False, "Недостаточно данных для прогнозирования"
            
            # Находим числовые колонки для прогнозирования
            numeric_cols = []
            for i, col in enumerate(self.df.columns):
                if i not in [year_col, month_col] and pd.api.types.is_numeric_dtype(self.df[col]):
                    if self.df[col].sum() > 0:  # Только колонки с положительными значениями
                        numeric_cols.append(i)
            
            if not numeric_cols:
                return False, "Не найдены числовые колонки для прогнозирования"
            
            print(f"  Найдено {len(numeric_cols)} числовых колонок для прогнозирования")
            
            # Создаем прогноз для каждой числовой колонки
            forecast_data = []
            
            for col_idx in numeric_cols:
                col_name = self.df.columns[col_idx]
                print(f"\n  📊 Прогнозирование для {col_name}:")
                
                # Подготавливаем данные
                forecast_result = self._create_forecast_for_column(
                    col_name, year_col_name, month_col_name, periods, method
                )
                
                if forecast_result:
                    forecast_data.append(forecast_result)
                    print(f"    ✅ Прогноз создан")
                else:
                    print(f"    ❌ Ошибка создания прогноза")
            
            if not forecast_data:
                return False, "Не удалось создать ни одного прогноза"
            
            # Сохраняем результаты
            self.forecast_results = {
                'forecast_data': forecast_data,
                'settings': config,
                'total_forecasts': len(forecast_data)
            }
            
            return True, f"Создано {len(forecast_data)} прогнозов"
            
        except Exception as e:
            return False, f"Ошибка в каскадном прогнозе: {str(e)}"
    
    def _create_forecast_for_column(self, col_name, year_col, month_col, periods, method):
        """Создание прогноза для конкретной колонки"""
        try:
            # Подготавливаем данные
            data = self.df[[year_col, month_col, col_name]].copy()
            data = data.dropna()
            
            if len(data) < 6:
                return None
            
            # Создаем временной индекс
            data['time_index'] = (data[year_col] - data[year_col].min()) * 12 + (data[month_col] - 1)
            
            # Сезонные признаки
            data['month_sin'] = np.sin(2 * np.pi * data[month_col] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data[month_col] / 12)
            
            # Квартальные признаки
            data['quarter'] = ((data[month_col] - 1) // 3) + 1
            for q in range(1, 5):
                data[f'q{q}'] = (data['quarter'] == q).astype(int)
            
            # Праздничные периоды
            data['holiday_period'] = (
                (data[month_col] == 12) |  # Декабрь
                (data[month_col] == 1) |   # Январь
                (data[month_col] == 2) |   # Февраль
                (data[month_col] == 3) |   # Март
                (data[month_col] == 5)     # Май
            ).astype(int)
            
            # Подготавливаем признаки
            features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
            X = data[features].fillna(0)
            y = data[col_name].fillna(0)
            
            # Выбираем модель
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = Ridge(alpha=1.0)
            
            # Обучаем модель
            model.fit(X, y)
            
            # Рассчитываем метрики
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Создаем прогноз
            last_year = data[year_col].max()
            last_month = data[month_col].max()
            last_time_index = data['time_index'].max()
            
            forecast_periods = []
            for i in range(1, periods + 1):
                period_data = {
                    'year': last_year + (i // 12),
                    'month': ((last_month + i - 1) % 12) + 1,
                    'time_index': last_time_index + i,
                    'month_sin': np.sin(2 * np.pi * (((last_month + i - 1) % 12) + 1) / 12),
                    'month_cos': np.cos(2 * np.pi * (((last_month + i - 1) % 12) + 1) / 12),
                }
                
                # Добавляем квартальные признаки
                month = period_data['month']
                quarter = ((month - 1) // 3) + 1
                for q in range(1, 5):
                    period_data[f'q{q}'] = 1 if quarter == q else 0
                
                # Праздничные периоды
                period_data['holiday_period'] = 1 if month in [12, 1, 2, 3, 5] else 0
                
                # Прогноз
                X_forecast = np.array([period_data[feature] for feature in features]).reshape(1, -1)
                forecast_value = model.predict(X_forecast)[0]
                forecast_value = max(0, forecast_value)  # Не допускаем отрицательные значения
                
                period_data['forecast'] = forecast_value
                forecast_periods.append(period_data)
            
            return {
                'column_name': col_name,
                'model_type': method,
                'r2': r2,
                'mae': mae,
                'forecast_periods': forecast_periods,
                'total_forecast': sum(p['forecast'] for p in forecast_periods)
            }
            
        except Exception as e:
            print(f"    Ошибка прогноза для {col_name}: {str(e)}")
            return None
    
    def save_results(self, session_id):
        """Сохранение результатов"""
        try:
            if not self.forecast_results:
                return None
            
            # Создаем DataFrame с результатами
            all_results = []
            
            for forecast in self.forecast_results['forecast_data']:
                for period in forecast['forecast_periods']:
                    all_results.append({
                    'column': forecast['column_name'],
                    'year': period['year'],
                    'month': period['month'],
                    'forecast': period['forecast'],
                    'model_type': forecast['model_type'],
                    'r2': forecast['r2'],
                    'mae': forecast['mae']
                    })
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                filename = f"cascaded_forecast_{session_id}.csv"
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                results_df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"💾 Результаты сохранены в {filepath}")
                return filepath
            
            return None
            
        except Exception as e:
            print(f"Ошибка сохранения результатов: {str(e)}")
            return None

# Глобальный объект приложения
forecast_app = WorkingForecastApp()

@app.route('/')
def index():
    """Главная страница - дашборд"""
    return render_template('dashboard.html', username='Пользователь', projects=[])

@app.route('/forecast')
def forecast():
    """Страница прогнозирования"""
    project_id = request.args.get('project')
    if project_id:
        # Загружаем проект
        try:
            project_file = os.path.join('projects', f"{project_id}.json")
            if os.path.exists(project_file):
                with open(project_file, 'r', encoding='utf-8') as f:
                    project = json.load(f)
                
                # Загружаем данные в forecast_app
                if project.get('data_info'):
                    # Проверяем, есть ли полные данные в data_info
                    if project['data_info'].get('full_data'):
                        # Используем полные данные из data_info
                        full_data = project['data_info']['full_data']
                    df = pd.DataFrame(full_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                elif project.get('processed_data') and project['processed_data'].get('sample_data'):
                    # Используем данные из processed_data
                    sample_data = project['processed_data']['sample_data']
                    df = pd.DataFrame(sample_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                else:
                    # Fallback: используем sample_data из data_info
                    sample_data = project['data_info'].get('sample_data', [])
                    if sample_data:
                        df = pd.DataFrame(sample_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                    print(f"DEBUG: Загружено {len(df)} строк в forecast_app для проекта {project_id}")
                    
                    # Обновляем время последнего доступа
                    project['updated_at'] = datetime.now().isoformat()
                    with open(project_file, 'w', encoding='utf-8') as f:
                        json.dump(project, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при загрузке проекта: {e}")
    
    return render_template('marfor_interface.html')

@app.route('/logout')
def logout():
    """Выход из системы"""
    return redirect('/')

@app.route('/favicon.ico')
def favicon():
    """Favicon"""
    return '', 204  # No Content

@app.route('/forecast/mapping')
def data_mapping():
    """Страница маппинга данных"""
    return render_template('data_mapping.html')

@app.route('/forecast/configure')
def forecast_configure():
    """Страница настройки прогноза"""
    return render_template('marfor_interface.html')

@app.route('/demo/mapping')
def demo_mapping():
    """Демо страница маппинга данных"""
    return render_template('demo_mapping.html')

@app.route('/api/apply_mapping', methods=['POST'])
def apply_mapping():
    """Применение маппинга данных"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        mapping_config = data.get('mapping')
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Применяем маппинг
        processed_data = forecast_app.apply_data_mapping(mapping_config)
        
        return jsonify({
            'success': True,
            'message': 'Маппинг применен успешно',
            'processed_data_info': {
                'shape': processed_data.shape,
                'columns': list(processed_data.columns),
                'dtypes': {col: str(dtype) for col, dtype in processed_data.dtypes.items()},
                'missing_values': processed_data.isnull().sum().to_dict(),
                'sample_data': convert_to_json_serializable(processed_data.head(5).fillna('').to_dict('records'))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при применении маппинга: {str(e)}'})

@app.route('/api/get_processed_data/<session_id>')
def get_processed_data(session_id):
    """Получение обработанных данных"""
    try:
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        # Получаем информацию об обработанных данных
        data_info = forecast_app.get_data_info()
        
        return jsonify({
            'success': True,
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении данных: {str(e)}'})

@app.route('/api/get_time_series_data/<session_id>')
def get_time_series_data(session_id):
    """Получение данных временных рядов для визуализации"""
    try:
        print(f"🔧 ВЕРСИЯ КОДА: 2.10.2 - Адаптивная ось Y с оптимальным масштабированием")
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        df = forecast_app.df.copy()
        
        # Отладочная информация
        print(f"DEBUG: Загружено {len(df)} строк данных")
        print(f"DEBUG: Колонки: {list(df.columns)}")
        
        # Получаем параметры из запроса
        time_column = request.args.get('time_column', '')
        metric_columns = request.args.getlist('metrics')
        slice_columns = request.args.getlist('slices')  # Добавляем поддержку срезов
        group_by = request.args.get('group_by', '')
        show_pivot = request.args.get('show_pivot', 'false').lower() == 'true'
        pivot_mode = request.args.get('pivot_mode', 'time-series')  # По умолчанию временные ряды
        split_by_slice = request.args.get('split_by_slice', '')  # Добавляем параметр разбивки по срезам
        
        print(f"DEBUG: time_column={time_column}, metrics={metric_columns}, group_by={group_by}, show_pivot={show_pivot}, pivot_mode={pivot_mode}, split_by_slice={split_by_slice}")
        print(f"DEBUG: Все параметры запроса: {dict(request.args)}")
        
        # Получаем маппинг из параметров запроса
        mapping_data = request.args.get('mapping_data', '{}')
        import json
        try:
            mapping_config = json.loads(mapping_data) if mapping_data else {}
        except json.JSONDecodeError as e:
            print(f"ERROR: Некорректный JSON в маппинге: {e}")
            return jsonify({
                'success': False, 
                'message': f'Критическая ошибка: некорректный формат маппинга данных. Ошибка JSON: {str(e)}'
            })
        
        print(f"DEBUG: Маппинг конфигурация: {mapping_config}")
        print(f"DEBUG: Количество колонок в маппинге: {len(mapping_config.get('columns', []))}")
        
        if not mapping_config or not mapping_config.get('columns'):
            print("ERROR: Маппинг не найден или пустой - это критическая ошибка!")
            return jsonify({
                'success': False, 
                'message': 'Критическая ошибка: маппинг данных не настроен или пустой. Пожалуйста, настройте маппинг колонок перед созданием сводной таблицы.'
            })
        
        if not time_column or not metric_columns:
            return jsonify({'success': False, 'message': 'Не указаны временная колонка или метрики'})
        
        # Проверяем существование колонок
        if time_column not in df.columns:
            return jsonify({'success': False, 'message': f'Временная колонка {time_column} не найдена'})
        
        for metric in metric_columns:
            if metric not in df.columns:
                return jsonify({'success': False, 'message': f'Метрика {metric} не найдена'})
        
        # Подготавливаем данные
        result_data = {
            'time_series': [],
            'grouped_series': {},
            'time_labels': [],
            'metrics': metric_columns,
            'pivot_table': None
        }
        
        print(f"DEBUG: Исходные данные содержат колонки: {df.columns.tolist()}")
        print(f"DEBUG: Первые 3 строки исходных данных:")
        for i, row in enumerate(df.head(3).to_dict('records')):
            print(f"  Строка {i}: {row}")
        
        # Сортируем по времени
        df_sorted = df.sort_values(time_column)
        
        # Получаем уникальные временные метки
        time_labels = df_sorted[time_column].unique()
        result_data['time_labels'] = [str(label) for label in time_labels]
        
        print(f"DEBUG: Найдено {len(time_labels)} уникальных временных меток: {time_labels[:10]}...")
        
        # Если есть группировка
        if group_by and group_by in df.columns:
            groups = df_sorted[group_by].unique()
            
            for group in groups:
                group_data = df_sorted[df_sorted[group_by] == group]
                group_series = {}
                
                for metric in metric_columns:
                    # Агрегируем данные по времени (сумма для числовых, последнее значение для категориальных)
                    if df[metric].dtype in ['int64', 'float64']:
                        metric_data = group_data.groupby(time_column)[metric].sum()
                else:
                    metric_data = group_data.groupby(time_column)[metric].last()
                    
                    # Заполняем пропуски
                    full_series = []
                    for time_label in time_labels:
                        if time_label in metric_data.index:
                            full_series.append(float(metric_data[time_label]) if pd.notna(metric_data[time_label]) else 0)
                        else:
                            full_series.append(0)
                    
                    group_series[metric] = full_series
                
                result_data['grouped_series'][str(group)] = group_series
        else:
            # Без группировки - общие данные
            for metric in metric_columns:
                if df[metric].dtype in ['int64', 'float64']:
                    metric_data = df_sorted.groupby(time_column)[metric].sum()
                else:
                    metric_data = df_sorted.groupby(time_column)[metric].last()
                
                # Заполняем пропуски
                full_series = []
                for time_label in time_labels:
                    if time_label in metric_data.index:
                        full_series.append(float(metric_data[time_label]) if pd.notna(metric_data[time_label]) else 0)
                    else:
                        full_series.append(0)
                
                result_data['time_series'].append({
                    'metric': metric,
                    'data': full_series
                })
        
        # Создаем сводную таблицу если запрошено
        print(f"DEBUG: show_pivot = {show_pivot}")
        if show_pivot:
            try:
                # Получаем настройки маппинга из sessionStorage (передаем через параметры)
                mapping_data = request.args.get('mapping_data', '{}')
                import json
                mapping = json.loads(mapping_data) if mapping_data else {}
                
                print(f"DEBUG: Получен маппинг: {mapping}")
                
                # Находим временные ряды с уровнями
                time_series_cols = []
                slice_cols = []
                if mapping.get('columns'):
                    for col in mapping['columns']:
                        if col.get('time_series') and col.get('nesting_level', 0) >= 0:
                            time_series_cols.append({
                                'name': col['name'],
                                'type': col['time_series'],
                                'level': col['nesting_level']
                            })
                        elif col.get('role') == 'dimension' and not col.get('time_series') and col.get('nesting_level', 0) >= 0:
                            slice_cols.append({
                                'name': col['name'],
                                'type': 'slice',
                                'level': col['nesting_level']
                            })
                
                # Сортируем по уровням
                time_series_cols.sort(key=lambda x: x['level'])
                slice_cols.sort(key=lambda x: x['level'])
                
                print(f"DEBUG: Временные ряды: {time_series_cols}")
                print(f"DEBUG: Срезы: {slice_cols}")
                
                # В зависимости от режима сводной таблицы
                print(f"DEBUG: pivot_mode = {pivot_mode}")
                print(f"DEBUG: split_by_slice = {split_by_slice}")
                
                if pivot_mode == 'time-series' and time_series_cols:
                    # В режиме временных рядов
                    print(f"DEBUG: Попадаем в блок time-series")
                    time_cols = time_series_cols.copy()
                    
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        print(f"DEBUG: Включаем режим разбивки по срезу: {split_by_slice}")
                        # Разбивка по срезу - временные колонки в строках, срез в столбцах
                        split_col = [col for col in slice_cols if col['name'] == split_by_slice][0]
                        print(f"DEBUG: Найден срез для разбивки: {split_col}")
                        
                        # Создаем сводную таблицу с разбивкой по срезу
                        pivot_cols = [col['name'] for col in time_cols]
                        print(f"DEBUG: Разбивка по срезу {split_by_slice}, временные колонки: {pivot_cols}")
                    else:
                        print(f"DEBUG: Обычный режим временных рядов без разбивки")
                        # Обычный режим временных рядов - только временные колонки
                        pivot_cols = [col['name'] for col in time_cols]
                        print(f"DEBUG: Временные колонки: {pivot_cols}")
                    
                    # Создаем pivot table с метриками
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        # С разбивкой по срезу
                        pivot_data = df_sorted.groupby(pivot_cols + [split_by_slice])[metric_columns].sum().reset_index()
                    else:
                        # Без разбивки - только временные колонки
                        pivot_data = df_sorted.groupby(pivot_cols)[metric_columns].sum().reset_index()
                    
                    # Создаем структуру с разбивкой по столбцам
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        unique_slices = sorted(pivot_data[split_by_slice].unique())
                        column_headers = {}
                        
                        for slice_value in unique_slices:
                            slice_data = pivot_data[pivot_data[split_by_slice] == slice_value]
                            column_headers[str(slice_value)] = {}
                            for metric in metric_columns:
                                column_headers[str(slice_value)][metric] = {}
                                for _, row in slice_data.iterrows():
                                    # Создаем ключ из временных значений
                                    time_key = '_'.join(str(row[col]) for col in pivot_cols)
                                    column_headers[str(slice_value)][metric][time_key] = float(row[metric]) if pd.notna(row[metric]) else 0
                    else:
                        # Без разбивки - простые заголовки
                        unique_slices = []
                        column_headers = {}
                    
                    # Включаем ВСЕ данные из маппинга для разбивки по срезам
                    all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                    available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                    
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': time_cols + [split_col],
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': split_by_slice,
                            'unique_slices': convert_to_json_serializable(unique_slices),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,
                            'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                        }
                    else:
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': time_cols,
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': '',
                            'unique_slices': convert_to_json_serializable(unique_slices),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,
                            'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                        }
                    
                    print(f"DEBUG: Создана сводная таблица с разбивкой:")
                    print(f"  - Колонки: {pivot_cols}")
                    print(f"  - Уникальные срезы: {unique_slices}")
                    print(f"  - Метрики: {metric_columns}")
                    print(f"  - Количество строк данных: {len(pivot_data)}")
                    print(f"  - Структура column_headers: {list(column_headers.keys())}")
                    print(f"  - Первые 3 строки данных:")
                    for i, row in enumerate(pivot_data.head(3).to_dict('records')):
                        print(f"    Строка {i}: {row}")
                    print(f"  - Структура данных: {pivot_data.columns.tolist()}")
                    print(f"  - Типы данных: {pivot_data.dtypes.to_dict()}")
                    
                    print(f"DEBUG: Создана сводная таблица с разбивкой по {split_by_slice}, уникальные значения: {unique_slices}")
                else:
                    # Обычный режим временных рядов - только временные колонки
                    all_cols = time_series_cols.copy()
                    print(f"DEBUG: Режим временных рядов - используем только временные колонки: {all_cols}")
                    
                    # Создаем сводную таблицу
                    pivot_cols = [col['name'] for col in all_cols]
                    print(f"DEBUG: Колонки для сводной таблицы: {pivot_cols}")
                    print(f"DEBUG: Метрики: {metric_columns}")
                    
                    pivot_data = df_sorted.groupby(pivot_cols)[metric_columns].sum().reset_index()
                    print(f"DEBUG: Создана сводная таблица с {len(pivot_data)} строками")
                    
                    # Форматируем данные для отображения
                    # Включаем ВСЕ данные из маппинга
                    all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                    available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                    
                    result_data['pivot_table'] = {
                        'columns': available_columns,
                        'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                        'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                        'time_series_info': all_cols,
                        'available_slices': slice_cols,
                        'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                    }
                
                if pivot_mode == 'slices':
                    # В режиме срезов - срезы в строках, метрики/временные ряды в столбцах
                    print(f"DEBUG: Попадаем в блок slices")
                    if split_by_slice and split_by_slice in [col['name'] for col in time_series_cols]:
                        print(f"DEBUG: Включаем режим разбивки по временному ряду: {split_by_slice}")
                        # Разбивка по временному ряду - срезы в строках, временной ряд в столбцах
                        slice_col_names = [col['name'] for col in slice_cols]
                        split_col = [col for col in time_series_cols if col['name'] == split_by_slice][0]
                        print(f"DEBUG: Найден временной ряд для разбивки: {split_col}")
                    
                        # Создаем pivot table с метриками в столбцах для каждого значения временного ряда
                        pivot_data = df_sorted.groupby(slice_col_names + [split_by_slice])[metric_columns].sum().reset_index()
                        
                        # Создаем структуру с разбивкой по столбцам (как в режиме временных рядов)
                        unique_time_values = sorted(pivot_data[split_by_slice].unique())
                        column_headers = {}
                        
                        for time_value in unique_time_values:
                            time_data = pivot_data[pivot_data[split_by_slice] == time_value]
                            column_headers[str(time_value)] = {}
                            for metric in metric_columns:
                                column_headers[str(time_value)][metric] = {}
                                for _, row in time_data.iterrows():
                                    # Создаем ключ из срезов (аналогично временным рядам)
                                    slice_key = '_'.join(str(row[col]) for col in slice_col_names)
                                    column_headers[str(time_value)][metric][slice_key] = float(row[metric]) if pd.notna(row[metric]) else 0
                        
                        # Включаем ВСЕ данные из маппинга для разбивки по временному ряду
                        all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                        available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                        
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': [],  # В режиме срезов временные ряды НЕ в строках
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': split_by_slice,
                            'unique_time_values': convert_to_json_serializable(unique_time_values),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,  # Срезы для строк
                            'available_time_series': time_series_cols,  # Временные ряды для разбивки по столбцам
                            'pivot_mode': 'slices'  # Явно указываем режим срезов
                        }
                    else:
                        # Обычный режим срезов без разбивки
                        print(f"DEBUG: Обычный режим срезов")
                        # Создаем сводную таблицу с срезами в строках, метриками в значениях
                        print(f"DEBUG: Все срезы: {slice_cols}")
                        print(f"DEBUG: Метрики: {metric_columns}")
                        
                        # Форматируем данные для отображения
                        # Включаем ВСЕ данные из маппинга
                        all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                        available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                        
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': [],  # В режиме срезов временные ряды не в строках
                            'available_slices': slice_cols,  # Срезы для строк
                            'available_time_series': time_series_cols,  # Временные ряды для разбивки
                            'metrics': metric_columns,  # Метрики для значений
                            'pivot_mode': 'slices'  # Явно указываем режим срезов
                        }
                        
                        print(f"DEBUG: Сводная таблица создана в режиме 'slices'")
                    
            except Exception as e:
                print(f"Ошибка создания сводной таблицы: {e}")
                result_data['pivot_table'] = None
        
        return jsonify({
            'success': True,
            'data': result_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении данных временных рядов: {str(e)}'})

@app.route('/api/save_project', methods=['POST'])
def save_project():
    """Сохранение проекта"""
    try:
        data = request.get_json()
        project_name = data.get('name', '')
        session_id = data.get('session_id', '')
        
        if not project_name:
            return jsonify({'success': False, 'message': 'Название проекта не указано'})
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Создаем объект проекта
        data_info = forecast_app.get_data_info()
        
        # Добавляем полные данные в data_info
        if forecast_app.df is not None:
            # Сохраняем все данные, а не только sample
            # Заменяем NaN на None для корректной JSON сериализации
            df_clean = forecast_app.df.fillna('')
            data_info['full_data'] = convert_to_json_serializable(df_clean.to_dict('records'))
        
        project = {
            'id': str(uuid.uuid4()),
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'session_id': session_id,
            'data_info': data_info,
            'data_mapping': data.get('data_mapping', {}),
            'processed_data': data.get('processed_data', {}),
            'status': 'saved'
        }
        
        # Сохраняем в файл
        projects_dir = 'projects'
        os.makedirs(projects_dir, exist_ok=True)
        
        project_file = os.path.join(projects_dir, f"{project['id']}.json")
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Проект сохранен успешно',
            'project_id': project['id']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при сохранении проекта: {str(e)}'})

@app.route('/api/load_project/<project_id>')
def load_project(project_id):
    """Загрузка проекта"""
    try:
        project_file = os.path.join('projects', f"{project_id}.json")
        
        if not os.path.exists(project_file):
            return jsonify({'success': False, 'message': 'Проект не найден'})
        
        with open(project_file, 'r', encoding='utf-8') as f:
            project = json.load(f)
        
        # Очищаем NaN значения в проекте
        def clean_nan_values(obj):
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, str) and obj in ['nan', 'NaN', 'null']:
                return ''
            elif pd.isna(obj) if hasattr(pd, 'isna') else False:
                return ''
            else:
                return obj
        
        project = clean_nan_values(project)
        
        # Обновляем время последнего доступа
        project['updated_at'] = datetime.now().isoformat()
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'project': project
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при загрузке проекта: {str(e)}'})

@app.route('/api/list_projects')
def list_projects():
    """Список сохраненных проектов"""
    try:
        projects_dir = 'projects'
        if not os.path.exists(projects_dir):
            return jsonify({'success': True, 'projects': []})
        
        projects = []
        for filename in os.listdir(projects_dir):
            if filename.endswith('.json'):
                project_file = os.path.join(projects_dir, filename)
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project = json.load(f)
                    # Возвращаем только основную информацию
                    projects.append({
                        'id': project['id'],
                    'name': project['name'],
                    'created_at': project['created_at'],
                    'updated_at': project['updated_at'],
                    'status': project.get('status', 'saved')
                    })
                except:
                    continue
        
        # Сортируем по времени обновления
        projects.sort(key=lambda x: x['updated_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'projects': projects
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении списка проектов: {str(e)}'})

@app.route('/api/delete_project/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Удаление проекта"""
    try:
        project_file = os.path.join('projects', f"{project_id}.json")
        
        if not os.path.exists(project_file):
            return jsonify({'success': False, 'message': 'Проект не найден'})
        
        os.remove(project_file)
        
        return jsonify({
            'success': True,
            'message': 'Проект удален успешно'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при удалении проекта: {str(e)}'})

@app.route('/old')
def old_interface():
    """Старый интерфейс"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Прогнозирование маркетинговых данных</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .main-content {
            padding: 40px;
        }

        .upload-section {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            border-color: #4285f4;
            background: #f0f7ff;
        }

        .upload-section.dragover {
            border-color: #4285f4;
            background: #e3f2fd;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: #4285f4;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-btn:hover {
            background: #3367d6;
            transform: translateY(-2px);
        }

        .settings-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .setting-group {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4285f4;
        }

        .setting-group h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }

        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
        }

        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4285f4;
            box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2);
        }

        .forecast-btn {
            background: linear-gradient(135deg, #34a853 0%, #137333 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-bottom: 20px;
        }

        .forecast-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(52, 168, 83, 0.3);
        }

        .forecast-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .results-section {
            margin-top: 30px;
            display: none;
        }

        .results-section.show {
            display: block;
        }

        .results-header {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .results-header h3 {
            color: #137333;
            margin-bottom: 10px;
        }

        .download-btn {
            background: #ff6b35;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            transition: all 0.3s ease;
        }

        .download-btn:hover {
            background: #e55a2b;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #4285f4;
        }

        .stat-card h4 {
            color: #666;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4285f4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #dc3545;
        }

        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #28a745;
        }

        .data-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .data-info h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }

        .column-mapping {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .mapping-group {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }

        .mapping-group h5 {
            color: #e65100;
            margin-bottom: 10px;
        }

        @media (max-width: 768px) {
            .main-content {
                padding: 20px;
            }
            
            .settings-section {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Прогнозирование маркетинговых данных</h1>
            <p>Создавайте точные прогнозы на основе исторических данных с помощью каскадной модели</p>
        </div>

        <div class="main-content">
            <!-- Загрузка файла -->
            <div class="upload-section" id="uploadSection">
                <h3>📁 Загрузите CSV файл с данными</h3>
                <p>Перетащите файл сюда или нажмите кнопку для выбора</p>
                <input type="file" id="fileInput" class="file-input" accept=".csv,.xlsx,.xls" />
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Выбрать файл
                </button>
                <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
                    Поддерживаются файлы CSV, Excel с колонками: год, месяц, числовые метрики
                </p>
            </div>

            <!-- Информация о данных -->
            <div class="data-info" id="dataInfo" style="display: none;">
                <h4>📊 Информация о загруженных данных</h4>
                <div id="dataInfoContent"></div>
            </div>

            <!-- Маппинг колонок -->
            <div class="column-mapping" id="columnMapping" style="display: none;">
                <div class="mapping-group">
                    <h5>🗓️ Колонка с годом</h5>
                    <select id="yearColumn" onchange="updateMapping()">
                    <option value="0">A (1-я колонка)</option>
                    </select>
                </div>
                <div class="mapping-group">
                    <h5>📅 Колонка с месяцем</h5>
                    <select id="monthColumn" onchange="updateMapping()">
                    <option value="1">B (2-я колонка)</option>
                    </select>
                </div>
            </div>

            <!-- Настройки прогноза -->
            <div class="settings-section">
                <div class="setting-group">
                    <h3>⚙️ Параметры прогноза</h3>
                    <div class="form-group">
                    <label for="periods">Количество периодов для прогноза:</label>
                    <input type="number" id="periods" value="4" min="1" max="12" />
                    </div>
                    <div class="form-group">
                    <label for="method">Метод прогнозирования:</label>
                    <select id="method">
                    <option value="random_forest">Random Forest (рекомендуется)</option>
                    <option value="linear">Линейная регрессия</option>
                    </select>
                    </div>
                </div>
            </div>

            <!-- Кнопка прогноза -->
            <button class="forecast-btn" id="forecastBtn" onclick="createForecast()" disabled>
                🔮 Создать прогноз
            </button>

            <!-- Загрузка -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Создание прогноза...</p>
            </div>

            <!-- Результаты -->
            <div class="results-section" id="resultsSection">
                <div class="results-header">
                    <h3>✅ Прогноз успешно создан!</h3>
                    <p>Результаты готовы для скачивания и анализа</p>
                    <button class="download-btn" onclick="downloadResults()">📥 Скачать CSV</button>
                </div>

                <div class="stats-grid" id="statsGrid">
                    <!-- Статистика будет добавлена динамически -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let dataInfo = null;

        // Обработка загрузки файла
        document.getElementById('fileInput').addEventListener('change', handleFileUpload);
        
        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            if (!file.name.toLowerCase().match(/\\.(csv|xlsx|xls)$/)) {
                showError('Пожалуйста, выберите CSV или Excel файл');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            showLoading(true);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    sessionId = data.session_id;
                    dataInfo = data.data_info;
                    showDataInfo(dataInfo);
                    showSuccess(data.message);
                    document.getElementById('forecastBtn').disabled = false;
                } else {
                    showError(data.message);
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Ошибка при загрузке файла: ' + error.message);
            });
        }

        function showDataInfo(info) {
            const dataInfoDiv = document.getElementById('dataInfo');
            const contentDiv = document.getElementById('dataInfoContent');
            
            let html = `
                <p><strong>Размер:</strong> ${info.shape[0]} строк, ${info.shape[1]} колонок</p>
                <p><strong>Колонки:</strong></p>
                <ul>
            `;
            
            info.columns.forEach((col, index) => {
                html += `<li>${index}: ${col} (${info.dtypes[col]})</li>`;
            });
            
            html += '</ul>';
            contentDiv.innerHTML = html;
            dataInfoDiv.style.display = 'block';
            
            // Обновляем селекты для маппинга
            updateColumnSelects();
        }

        function updateColumnSelects() {
            const yearSelect = document.getElementById('yearColumn');
            const monthSelect = document.getElementById('monthColumn');
            
            // Очищаем селекты
            yearSelect.innerHTML = '';
            monthSelect.innerHTML = '';
            
            // Добавляем опции
            dataInfo.columns.forEach((col, index) => {
                const option1 = document.createElement('option');
                option1.value = index;
                option1.textContent = `${String.fromCharCode(65 + index)} (${index + 1}-я колонка): ${col}`;
                yearSelect.appendChild(option1);
                
                const option2 = document.createElement('option');
                option2.value = index;
                option2.textContent = `${String.fromCharCode(65 + index)} (${index + 1}-я колонка): ${col}`;
                monthSelect.appendChild(option2);
            });
            
            // Устанавливаем значения по умолчанию
            yearSelect.value = '0';
            monthSelect.value = '1';
            
            document.getElementById('columnMapping').style.display = 'grid';
        }

        function updateMapping() {
            // Функция для обновления маппинга (можно расширить)
        }

        function createForecast() {
            if (!sessionId) {
                showError('Сначала загрузите файл с данными');
                return;
            }

            const settings = {
                periods: parseInt(document.getElementById('periods').value),
                method: document.getElementById('method').value,
                year_column: parseInt(document.getElementById('yearColumn').value),
                month_column: parseInt(document.getElementById('monthColumn').value)
            };

            showLoading(true);

            fetch('/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    showResults(data);
                    showSuccess(data.message);
                } else {
                    showError(data.message);
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Ошибка при создании прогноза: ' + error.message);
            });
        }

        function showResults(data) {
            const section = document.getElementById('resultsSection');
            section.classList.add('show');
            
            // Показываем статистику
            showStats(data);
        }

        function showStats(data) {
            const statsGrid = document.getElementById('statsGrid');
            
            let html = `
                <div class="stat-card">
                    <h4>Создано прогнозов</h4>
                    <div class="value">${data.total_forecasts}</div>
                </div>
                <div class="stat-card">
                    <h4>Периодов прогноза</h4>
                    <div class="value">${data.settings.periods}</div>
                </div>
                <div class="stat-card">
                    <h4>Метод</h4>
                    <div class="value">${data.settings.method === 'random_forest' ? 'Random Forest' : 'Линейный'}</div>
                </div>
            `;
            
            statsGrid.innerHTML = html;
        }

        function downloadResults() {
            if (!sessionId) return;
            
            window.open(`/download/${sessionId}`, '_blank');
        }

        function showLoading(show) {
            const loading = document.getElementById('loading');
            const btn = document.getElementById('forecastBtn');
            
            if (show) {
                loading.classList.add('show');
                btn.disabled = true;
            } else {
                loading.classList.remove('show');
                btn.disabled = false;
            }
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            
            const container = document.querySelector('.main-content');
            container.insertBefore(errorDiv, container.firstChild);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }

        function showSuccess(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'success';
            successDiv.textContent = message;
            
            const container = document.querySelector('.main-content');
            container.insertBefore(successDiv, container.firstChild);
            
            setTimeout(() => {
                successDiv.remove();
            }, 3000);
        }
    </script>
</body>
</html>
    """)

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
        success, message = forecast_app.load_data_from_file(filepath)
        
        if success:
            forecast_app.session_id = session_id
            data_info = forecast_app.get_data_info()
            
            # Очищаем NaN значения для JSON
            import json
            data_info_json = json.dumps(data_info, default=str)
            data_info_clean = json.loads(data_info_json)
            
            return jsonify({
                'success': True, 
                'message': message,
                'session_id': session_id,
                'data_info': data_info_clean
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Недопустимый тип файла'})

@app.route('/forecast_api', methods=['POST'])
def forecast_api():
    """Создание прогноза"""
    try:
        config = request.json
        print(f"DEBUG: Получен запрос на прогнозирование: {config}")
        
        # Получаем session_id из запроса
        session_id = config.get('session_id')
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Проверяем наличие данных
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        # Получаем настройки маппинга из запроса или используем значения по умолчанию
        mapping_data = config.get('mapping_data')
        if mapping_data:
            mapping = json.loads(mapping_data)
            print(f"DEBUG: Используем маппинг из запроса: {mapping}")
        else:
            # Используем значения по умолчанию
            mapping = {
                'year': config.get('year_column', 0),
                'month': config.get('month_column', 1)
            }
        
        # Устанавливаем маппинг колонок
        forecast_app.set_data_mapping(mapping)
        
        # Подготавливаем конфигурацию для прогноза
        forecast_config = {
            'periods': config.get('periods', 4),
            'method': config.get('method', 'random_forest'),
            'target_metric': config.get('target_metric'),
            'enable_cascade': config.get('enable_cascade', True)
        }
        
        print(f"DEBUG: Конфигурация прогноза: {forecast_config}")
        
        # Запускаем прогноз
        success, message = forecast_app.run_cascaded_forecast(forecast_config)
        
        if success:
            # Сохраняем результаты
            forecast_app.save_results(forecast_app.session_id)
            
            return jsonify({
                'success': True,
                'message': message,
                'total_forecasts': forecast_app.forecast_results.get('total_forecasts', 0),
                'settings': forecast_config
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        print(f"ERROR: Ошибка при выполнении прогноза: {e}")
        return jsonify({'success': False, 'message': f'Ошибка сервера: {str(e)}'})

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
    print("🚀 Запуск MARFOR веб-приложения...")
    print("📊 Каскадная модель с Random Forest")
    print("🔧 ВЕРСИЯ КОДА: 2.10.2 - Адаптивная ось Y с оптимальным масштабированием")
    print("🌐 Откройте http://localhost:5001 в браузере")
    app.run(debug=True, host='0.0.0.0', port=5001)
