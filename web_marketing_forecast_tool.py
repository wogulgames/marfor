#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для универсального инструмента прогнозирования маркетинга
Flask приложение с возможностью загрузки файлов и настройки параметров
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Flask
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import tempfile
import uuid
from datetime import datetime

# Scikit-learn модели
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
app.secret_key = 'your-secret-key-here'

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

class WebMarketingForecastTool:
    def __init__(self):
        """Инициализация веб-инструмента прогнозирования"""
        self.df = None
        self.config = {}
        self.channel_dependencies = {}
        self.models = {}
        self.forecasts = {}
        self.simulation_results = {}
        self.budget_impact_models = {}
        self.session_id = None
        
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
    
    def setup_configuration(self, config_data: dict):
        """Настройка конфигурации"""
        self.config = config_data
        
        # Определяем каналы
        self._identify_channels()
        
        return True, "Конфигурация успешно применена"
    
    def _identify_channels(self):
        """Определение каналов на основе конфигурации"""
        if not self.config or 'channels' not in self.config:
            return
        
        # Создаем поле channel_type
        self.df['channel_type'] = 'other'
        
        for channel_name, channel_config in self.config['channels'].items():
            filter_field = channel_config['filter_field']
            filter_values = channel_config['filter_values']
            
            if filter_field in self.df.columns:
                mask = self.df[filter_field].isin(filter_values)
                self.df.loc[mask, 'channel_type'] = channel_name
    
    def analyze_channel_dependencies(self):
        """Анализ зависимостей между каналами"""
        if not self.config.get('dependencies', {}).get('enabled', False):
            return {}
        
        dependencies = {}
        
        # Находим платные каналы
        paid_channels = [name for name, config in self.config['channels'].items() if config['is_paid']]
        
        for channel_name in paid_channels:
            channel_config = self.config['channels'][channel_name]
            budget_field = channel_config.get('budget_field')
            
            if not budget_field or budget_field not in self.df.columns:
                continue
            
            channel_data = self.df[self.df['channel_type'] == channel_name]
            
            if len(channel_data) < 10:
                continue
            
            dependencies[channel_name] = {}
            
            # Анализируем влияние бюджета на трафик
            traffic_fields = ['first_traffic', 'repeat_traffic', 'traffic_total']
            
            for traffic_field in traffic_fields:
                if traffic_field in channel_data.columns:
                    correlation = channel_data[budget_field].corr(channel_data[traffic_field])
                    dependencies[channel_name][traffic_field] = {
                        'budget_correlation': correlation,
                        'data_points': len(channel_data)
                    }
        
        self.channel_dependencies = dependencies
        return dependencies
    
    def build_models(self):
        """Построение моделей"""
        if not self.config or 'channels' not in self.config:
            return False, "Конфигурация не настроена"
        
        # Подготавливаем данные для обучения
        train_data = self._prepare_training_data()
        
        models = {}
        
        # Строим модели для каждого канала
        for channel_name, channel_config in self.config['channels'].items():
            channel_data = train_data[train_data['channel_type'] == channel_name]
            
            if len(channel_data) == 0:
                continue
            
            channel_models = {}
            
            for metric_name, metric_field in self.config['metrics'].items():
                if metric_field in channel_data.columns:
                    features = self._prepare_features(channel_data)
                    target = channel_data[metric_field]
                    
                    model = self._train_model(features, target, metric_name)
                    
                    if model:
                        channel_models[metric_name] = model
            
            if channel_models:
                models[channel_name] = channel_models
        
        self.models = models
        
        # Строим модели влияния бюджета
        self._build_budget_impact_models()
        
        return True, f"Построено {len(models)} моделей каналов"
    
    def _prepare_training_data(self):
        """Подготовка данных для обучения"""
        time_fields = list(self.config['time_fields'].values())
        if 'year' in time_fields and 'month' in time_fields:
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
    
    def _prepare_features(self, data):
        """Подготовка признаков для модели"""
        features_df = data.copy()
        
        # Временные признаки
        time_fields = list(self.config['time_fields'].values())
        if 'year' in features_df.columns and 'month' in features_df.columns:
            features_df['time_index'] = (features_df['year'] - features_df['year'].min()) * 12 + (features_df['month'] - 1)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            
            # Квартальные признаки
            features_df['quarter'] = ((features_df['month'] - 1) // 3) + 1
            for q in range(1, 5):
                features_df[f'q{q}'] = (features_df['quarter'] == q).astype(int)
        
        # Категориальные признаки
        for dim_name, dim_field in self.config['dimensions'].items():
            if dim_field in features_df.columns:
                le = LabelEncoder()
                features_df[f'{dim_name}_encoded'] = le.fit_transform(features_df[dim_field].astype(str))
        
        return features_df
    
    def _train_model(self, features, target, metric_name):
        """Обучение модели"""
        try:
            numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
            X = features[numeric_features].fillna(0)
            y = target.fillna(0)
            
            if len(X) == 0 or len(y) == 0:
                return None
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            return {
                'model': model,
                'scaler': scaler,
                'features': numeric_features,
                'r2': r2,
                'mae': mae
            }
            
        except Exception as e:
            print(f"Ошибка в обучении модели: {str(e)}")
            return None
    
    def _build_budget_impact_models(self):
        """Построение моделей влияния бюджета"""
        paid_channels = [name for name, config in self.config['channels'].items() if config['is_paid']]
        
        for channel_name in paid_channels:
            channel_config = self.config['channels'][channel_name]
            budget_field = channel_config.get('budget_field')
            
            if not budget_field or budget_field not in self.df.columns:
                continue
            
            channel_data = self.df[self.df['channel_type'] == channel_name]
            
            if len(channel_data) > 20:
                traffic_fields = ['first_traffic', 'repeat_traffic', 'traffic_total']
                
                for traffic_field in traffic_fields:
                    if traffic_field in channel_data.columns:
                        X = channel_data[[budget_field]].fillna(0)
                        y = channel_data[traffic_field].fillna(0)
                        
                        if len(X) > 10:
                            model = LinearRegression()
                            model.fit(X, y)
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)
                            
                            model_key = f'{channel_name}_{budget_field}_to_{traffic_field}'
                            self.budget_impact_models[model_key] = {
                                'model': model,
                                'r2': r2,
                                'coefficient': model.coef_[0],
                                'intercept': model.intercept_
                            }
    
    def create_forecast(self, forecast_periods=16):
        """Создание прогноза"""
        if not self.models:
            return False, "Нет обученных моделей"
        
        forecasts = {}
        
        for channel_name, channel_models in self.models.items():
            channel_forecast = {}
            
            for metric_name, model_info in channel_models.items():
                future_data = self._create_future_periods(forecast_periods)
                future_features = self._prepare_features(future_data)
                
                forecast_values = self._make_prediction(model_info, future_features)
                
                channel_forecast[metric_name] = {
                    'values': forecast_values,
                    'periods': future_data[['year', 'month']].to_dict('records')
                }
            
            forecasts[channel_name] = channel_forecast
        
        self.forecasts = forecasts
        return True, f"Создан прогноз для {len(forecasts)} каналов"
    
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
    
    def _make_prediction(self, model_info, features):
        """Создание прогноза"""
        try:
            model_features = model_info['features']
            X = features[model_features].fillna(0)
            X_scaled = model_info['scaler'].transform(X)
            predictions = model_info['model'].predict(X_scaled)
            return np.maximum(predictions, 0)
        except Exception as e:
            print(f"Ошибка в прогнозе: {str(e)}")
            return np.zeros(len(features))
    
    def simulate_budget_scenarios(self, scenarios):
        """Моделирование сценариев бюджета"""
        if not self.forecasts:
            return False, "Нет прогнозов для моделирования"
        
        simulation_results = {}
        
        for i, scenario in enumerate(scenarios):
            modified_forecast = self._apply_budget_changes(scenario)
            impact_analysis = self._analyze_budget_impact(scenario, modified_forecast)
            
            simulation_results[f"scenario_{i+1}"] = {
                'scenario': scenario,
                'modified_forecast': modified_forecast,
                'impact_analysis': impact_analysis
            }
        
        self.simulation_results = simulation_results
        return True, f"Выполнено {len(scenarios)} сценариев моделирования"
    
    def _apply_budget_changes(self, scenario):
        """Применение изменений бюджета"""
        modified_forecast = {}
        
        for channel_name, channel_forecast in self.forecasts.items():
            modified_channel = {}
            
            for metric_name, metric_forecast in channel_forecast.items():
                values = metric_forecast['values'].copy()
                
                channel_config = self.config['channels'][channel_name]
                budget_field = channel_config.get('budget_field')
                
                if budget_field and metric_name == budget_field and 'budget_change' in scenario:
                    change_factor = 1 + scenario['budget_change'] / 100
                    values = values * change_factor
                
                if 'budget_change' in scenario:
                    model_key = f'{channel_name}_{budget_field}_to_{metric_name}'
                    if model_key in self.budget_impact_models:
                        model_info = self.budget_impact_models[model_key]
                        budget_change = scenario['budget_change'] / 100
                        traffic_change = budget_change * model_info['coefficient']
                        values = values * (1 + traffic_change)
                
                modified_channel[metric_name] = {
                    'values': values,
                    'periods': metric_forecast['periods']
                }
            
            modified_forecast[channel_name] = modified_channel
        
        return modified_forecast
    
    def _analyze_budget_impact(self, scenario, modified_forecast):
        """Анализ влияния изменений бюджета"""
        impact_analysis = {}
        
        for channel_name in self.forecasts.keys():
            if channel_name in modified_forecast:
                for metric_name in self.forecasts[channel_name].keys():
                    if metric_name in modified_forecast[channel_name]:
                        original = np.sum(self.forecasts[channel_name][metric_name]['values'])
                        modified = np.sum(modified_forecast[channel_name][metric_name]['values'])
                        
                        if original > 0:
                            change_pct = (modified - original) / original * 100
                            impact_analysis[f"{channel_name}_{metric_name}"] = change_pct
        
        # Общее влияние на выручку
        total_original_revenue = 0
        total_modified_revenue = 0
        
        for channel_name in self.forecasts.keys():
            if 'revenue' in self.forecasts[channel_name]:
                total_original_revenue += np.sum(self.forecasts[channel_name]['revenue']['values'])
            if channel_name in modified_forecast and 'revenue' in modified_forecast[channel_name]:
                total_modified_revenue += np.sum(modified_forecast[channel_name]['revenue']['values'])
        
        if total_original_revenue > 0:
            impact_analysis['revenue_impact'] = (total_modified_revenue - total_original_revenue) / total_original_revenue * 100
        
        return impact_analysis
    
    def generate_visualization(self):
        """Генерация визуализации"""
        if not self.forecasts:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Прогноз маркетинговых метрик', fontsize=16)
        
        # График 1: Общий прогноз по каналам
        ax1 = axes[0, 0]
        channel_totals = {}
        for channel_name, channel_forecast in self.forecasts.items():
            if 'revenue' in channel_forecast:
                total = np.sum(channel_forecast['revenue']['values'])
                channel_totals[channel_name] = total
        
        if channel_totals:
            ax1.pie(channel_totals.values(), labels=channel_totals.keys(), autopct='%1.1f%%')
            ax1.set_title('Распределение выручки по каналам')
        
        # График 2: Тренд выручки
        ax2 = axes[0, 1]
        for channel_name, channel_forecast in self.forecasts.items():
            if 'revenue' in channel_forecast:
                periods = channel_forecast['revenue']['periods']
                values = channel_forecast['revenue']['values']
                
                months = [f"{p['year']}-{p['month']:02d}" for p in periods]
                ax2.plot(months, values, label=channel_name, marker='o')
        
        ax2.set_title('Тренд выручки по каналам')
        ax2.set_xlabel('Период')
        ax2.set_ylabel('Выручка')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # График 3: Влияние бюджета
        ax3 = axes[1, 0]
        if self.simulation_results:
            scenarios = []
            revenue_impacts = []
            
            for scenario_name, results in self.simulation_results.items():
                scenarios.append(results['scenario'].get('name', scenario_name))
                revenue_impacts.append(results['impact_analysis'].get('revenue_impact', 0))
            
            ax3.bar(scenarios, revenue_impacts)
            ax3.set_title('Влияние изменений бюджета на выручку')
            ax3.set_ylabel('Изменение выручки (%)')
            ax3.tick_params(axis='x', rotation=45)
        
        # График 4: Корреляции
        ax4 = axes[1, 1]
        if self.channel_dependencies:
            correlations = []
            labels = []
            
            for channel, deps in self.channel_dependencies.items():
                for metric, metric_deps in deps.items():
                    if isinstance(metric_deps, dict) and 'budget_correlation' in metric_deps:
                        correlations.append(metric_deps['budget_correlation'])
                        labels.append(f"{channel}\n{metric}")
            
            if correlations:
                ax4.bar(range(len(correlations)), correlations)
                ax4.set_title('Корреляции между бюджетом и трафиком')
                ax4.set_ylabel('Корреляция')
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Сохраняем график в base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def save_results(self, session_id):
        """Сохранение результатов"""
        if not self.forecasts:
            return None
        
        # Собираем все прогнозы
        all_results = []
        
        for channel_name, channel_forecast in self.forecasts.items():
            for metric_name, metric_forecast in channel_forecast.items():
                for i, (period, value) in enumerate(zip(metric_forecast['periods'], metric_forecast['values'])):
                    all_results.append({
                        'channel': channel_name,
                        'metric': metric_name,
                        'year': period['year'],
                        'month': period['month'],
                        'forecast_value': value
                    })
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(all_results)
        filename = f"forecast_results_{session_id}.csv"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        results_df.to_csv(filepath, index=False)
        
        return filepath

# Глобальный объект инструмента
forecast_tool = WebMarketingForecastTool()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

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
        success, message = forecast_tool.load_data_from_file(filepath)
        
        if success:
            forecast_tool.session_id = session_id
            data_info = forecast_tool.get_data_info()
            return jsonify({
                'success': True, 
                'message': message,
                'session_id': session_id,
                'data_info': data_info
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Недопустимый тип файла'})

@app.route('/configure', methods=['POST'])
def configure():
    """Настройка конфигурации"""
    config_data = request.json
    
    success, message = forecast_tool.setup_configuration(config_data)
    
    if success:
        # Анализируем зависимости
        dependencies = forecast_tool.analyze_channel_dependencies()
        
        return jsonify({
            'success': True,
            'message': message,
            'dependencies': dependencies
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/build_models', methods=['POST'])
def build_models():
    """Построение моделей"""
    success, message = forecast_tool.build_models()
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'models_count': len(forecast_tool.models)
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/forecast', methods=['POST'])
def create_forecast():
    """Создание прогноза"""
    forecast_periods = request.json.get('periods', 16)
    
    success, message = forecast_tool.create_forecast(forecast_periods)
    
    if success:
        # Генерируем визуализацию
        visualization = forecast_tool.generate_visualization()
        
        return jsonify({
            'success': True,
            'message': message,
            'visualization': visualization,
            'forecasts': forecast_tool.forecasts
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/simulate', methods=['POST'])
def simulate():
    """Моделирование сценариев"""
    scenarios = request.json.get('scenarios', [])
    
    success, message = forecast_tool.simulate_budget_scenarios(scenarios)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'simulation_results': forecast_tool.simulation_results
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/download/<session_id>')
def download_results(session_id):
    """Скачивание результатов"""
    filename = f"forecast_results_{session_id}.csv"
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        return jsonify({'success': False, 'message': 'Файл не найден'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
