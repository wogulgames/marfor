#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пользовательски настраиваемый инструмент для прогнозирования и бюджетирования маркетинга
Пользователь сам определяет каналы, поля и зависимости
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any

class UserConfigurableMarketingTool:
    def __init__(self):
        """Инициализация пользовательски настраиваемого инструмента"""
        self.df = None
        self.config = {}
        self.channel_dependencies = {}
        self.models = {}
        self.forecasts = {}
        self.simulation_results = {}
        self.budget_impact_models = {}
        
    def setup_configuration_interactive(self):
        """Интерактивная настройка конфигурации"""
        print("🔧 ИНТЕРАКТИВНАЯ НАСТРОЙКА КОНФИГУРАЦИИ:")
        print("=" * 50)
        
        # Загружаем данные для анализа полей
        if self.df is None:
            print("❌ Сначала загрузите данные с помощью load_data()")
            return
        
        print(f"📊 Доступные поля в данных:")
        for i, col in enumerate(self.df.columns):
            print(f"  {i+1}. {col}")
        
        # Настройка временных полей
        print(f"\n⏰ НАСТРОЙКА ВРЕМЕННЫХ ПОЛЕЙ:")
        year_field = input("Введите название поля для года (по умолчанию 'year'): ").strip() or 'year'
        month_field = input("Введите название поля для месяца (по умолчанию 'month'): ").strip() or 'month'
        
        # Настройка измерений
        print(f"\n📏 НАСТРОЙКА ИЗМЕРЕНИЙ (dimensions):")
        dimensions = {}
        while True:
            dim_name = input("Введите название измерения (или 'done' для завершения): ").strip()
            if dim_name.lower() == 'done':
                break
            if dim_name in self.df.columns:
                dimensions[dim_name] = dim_name
                print(f"  ✅ Добавлено измерение: {dim_name}")
            else:
                print(f"  ❌ Поле '{dim_name}' не найдено в данных")
        
        # Настройка метрик
        print(f"\n📈 НАСТРОЙКА МЕТРИК:")
        metrics = {}
        while True:
            metric_name = input("Введите название метрики (или 'done' для завершения): ").strip()
            if metric_name.lower() == 'done':
                break
            if metric_name in self.df.columns:
                metrics[metric_name] = metric_name
                print(f"  ✅ Добавлена метрика: {metric_name}")
            else:
                print(f"  ❌ Поле '{metric_name}' не найдено в данных")
        
        # Настройка каналов
        print(f"\n📺 НАСТРОЙКА КАНАЛОВ:")
        channels = {}
        while True:
            channel_name = input("Введите название канала (или 'done' для завершения): ").strip()
            if channel_name.lower() == 'done':
                break
            
            # Определение типа канала
            is_paid = input(f"Является ли канал '{channel_name}' платным? (y/n): ").strip().lower() == 'y'
            
            # Поле для фильтрации канала
            filter_field = input(f"Введите поле для фильтрации канала '{channel_name}': ").strip()
            if filter_field not in self.df.columns:
                print(f"  ❌ Поле '{filter_field}' не найдено в данных")
                continue
            
            # Значения для фильтрации
            print(f"Доступные значения в поле '{filter_field}':")
            unique_values = self.df[filter_field].unique()[:10]  # Показываем первые 10
            for i, val in enumerate(unique_values):
                print(f"  {i+1}. {val}")
            
            filter_values_input = input(f"Введите значения для канала '{channel_name}' (через запятую): ").strip()
            filter_values = [v.strip() for v in filter_values_input.split(',')]
            
            # Поле бюджета для платных каналов
            budget_field = None
            if is_paid:
                budget_field = input(f"Введите поле бюджета для канала '{channel_name}': ").strip()
                if budget_field not in self.df.columns:
                    print(f"  ❌ Поле '{budget_field}' не найдено в данных")
                    budget_field = None
            
            channels[channel_name] = {
                'is_paid': is_paid,
                'filter_field': filter_field,
                'filter_values': filter_values,
                'budget_field': budget_field
            }
            
            print(f"  ✅ Добавлен канал: {channel_name} ({'платный' if is_paid else 'органический'})")
        
        # Настройка зависимостей
        print(f"\n🔗 НАСТРОЙКА ЗАВИСИМОСТЕЙ:")
        dependencies = {}
        
        if any(ch['is_paid'] for ch in channels.values()):
            enable_dependencies = input("Включить анализ зависимостей между каналами? (y/n): ").strip().lower() == 'y'
            if enable_dependencies:
                dependencies['enabled'] = True
                cross_influence = input("Введите коэффициент кросс-канального влияния (0.0-1.0, по умолчанию 0.1): ").strip()
                dependencies['cross_channel_influence'] = float(cross_influence) if cross_influence else 0.1
            else:
                dependencies['enabled'] = False
        else:
            dependencies['enabled'] = False
        
        # Создаем конфигурацию
        self.config = {
            "time_fields": {
                "year": year_field,
                "month": month_field
            },
            "dimensions": dimensions,
            "metrics": metrics,
            "channels": channels,
            "dependencies": dependencies
        }
        
        print(f"\n✅ Конфигурация создана!")
        self._print_config_summary()
    
    def setup_configuration_from_dict(self, config_dict: Dict):
        """Настройка конфигурации из словаря"""
        print("🔧 НАСТРОЙКА КОНФИГУРАЦИИ ИЗ СЛОВАРЯ:")
        
        self.config = config_dict
        print("✅ Конфигурация загружена")
        self._print_config_summary()
    
    def setup_default_configuration(self):
        """Настройка конфигурации по умолчанию для текущего датасета"""
        print("🔧 НАСТРОЙКА КОНФИГУРАЦИИ ПО УМОЛЧАНИЮ:")
        
        self.config = {
            "time_fields": {
                "year": "year",
                "month": "month"
            },
            "dimensions": {
                "region": "region_to",
                "subdivision": "subdivision", 
                "category": "category"
            },
            "metrics": {
                "revenue": "revenue_total",
                "traffic": "traffic_total",
                "transactions": "transacitons_total",
                "ads_cost": "ads_cost",
                "first_traffic": "first_traffic",
                "repeat_traffic": "repeat_traffic",
                "first_transactions": "first_transactions",
                "repeat_transactions": "repeat_transactions"
            },
            "channels": {
                "paid_apps": {
                    "is_paid": True,
                    "filter_field": "subdivision",
                    "filter_values": ["paid_apps"],
                    "budget_field": "ads_cost"
                },
                "paid_web": {
                    "is_paid": True,
                    "filter_field": "subdivision",
                    "filter_values": ["paid_web"],
                    "budget_field": "ads_cost"
                },
                "organic_apps": {
                    "is_paid": False,
                    "filter_field": "subdivision",
                    "filter_values": ["organic_apps"],
                    "budget_field": None
                },
                "organic_web": {
                    "is_paid": False,
                    "filter_field": "subdivision",
                    "filter_values": ["organic_web"],
                    "budget_field": None
                },
                "brand": {
                    "is_paid": False,
                    "filter_field": "subdivision",
                    "filter_values": ["brand"],
                    "budget_field": None
                }
            },
            "dependencies": {
                "enabled": True,
                "cross_channel_influence": 0.1
            }
        }
        
        print("✅ Конфигурация по умолчанию загружена")
        self._print_config_summary()
    
    def _print_config_summary(self):
        """Вывод сводки конфигурации"""
        print(f"\n📋 СВОДКА КОНФИГУРАЦИИ:")
        print(f"  Временные поля: {list(self.config['time_fields'].values())}")
        print(f"  Измерения: {list(self.config['dimensions'].values())}")
        print(f"  Метрики: {list(self.config['metrics'].values())}")
        print(f"  Каналы: {list(self.config['channels'].keys())}")
        print(f"  Зависимости: {'Включены' if self.config['dependencies']['enabled'] else 'Отключены'}")
        
        # Детали по каналам
        print(f"\n  📺 ДЕТАЛИ КАНАЛОВ:")
        for channel_name, channel_config in self.config['channels'].items():
            channel_type = "платный" if channel_config['is_paid'] else "органический"
            budget_field = channel_config.get('budget_field', 'нет')
            print(f"    {channel_name}: {channel_type}, бюджет: {budget_field}")
    
    def load_data(self, csv_file: str):
        """Загрузка данных"""
        print(f"\n📁 ЗАГРУЗКА ДАННЫХ ИЗ {csv_file}:")
        
        # Пробуем разные разделители
        separators = [',', ';', '\t', '|']
        for sep in separators:
            try:
                self.df = pd.read_csv(csv_file, sep=sep)
                if len(self.df.columns) > 1:
                    print(f"✅ Загружено с разделителем '{sep}': {len(self.df)} записей, {len(self.df.columns)} колонок")
                    break
            except:
                continue
        
        if self.df is None or len(self.df.columns) <= 1:
            print("❌ Не удалось загрузить файл с правильным разделителем")
            return False
        
        # Очистка данных
        self._clean_data()
        return True
    
    def _clean_data(self):
        """Очистка данных"""
        print(f"\n🧹 ОЧИСТКА ДАННЫХ:")
        
        initial_count = len(self.df)
        
        # Очистка временных полей (если конфигурация уже настроена)
        if self.config:
            for field in self.config['time_fields'].values():
                if field in self.df.columns:
                    self.df[field] = pd.to_numeric(self.df[field], errors='coerce')
            
            # Удаляем строки с пустыми временными данными
            time_cols = list(self.config['time_fields'].values())
            self.df = self.df.dropna(subset=time_cols)
            
            # Очистка метрик
            for metric in self.config['metrics'].values():
                if metric in self.df.columns:
                    self.df[metric] = self.df[metric].astype(str).str.replace(',', '').str.replace(' ', '')
                    self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
                    self.df[metric] = self.df[metric].fillna(0)
        
        print(f"  Удалено {initial_count - len(self.df)} записей с пустыми данными")
        print(f"  После очистки: {len(self.df)} записей")
    
    def identify_channels(self):
        """Определение каналов на основе пользовательской конфигурации"""
        print(f"\n🔍 ОПРЕДЕЛЕНИЕ КАНАЛОВ:")
        
        if not self.config or 'channels' not in self.config:
            print("❌ Конфигурация каналов не настроена")
            return
        
        # Создаем поле channel_type
        self.df['channel_type'] = 'other'
        
        for channel_name, channel_config in self.config['channels'].items():
            filter_field = channel_config['filter_field']
            filter_values = channel_config['filter_values']
            
            if filter_field in self.df.columns:
                mask = self.df[filter_field].isin(filter_values)
                self.df.loc[mask, 'channel_type'] = channel_name
        
        # Анализ распределения каналов
        channel_counts = self.df['channel_type'].value_counts()
        print(f"  📊 Распределение каналов:")
        for channel, count in channel_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"    {channel}: {count} записей ({percentage:.1f}%)")
    
    def analyze_channel_dependencies(self):
        """Анализ зависимостей между каналами"""
        print(f"\n🔍 АНАЛИЗ ЗАВИСИМОСТЕЙ МЕЖДУ КАНАЛАМИ:")
        
        if not self.config['dependencies']['enabled']:
            print("  ⚠️  Анализ зависимостей отключен в конфигурации")
            return
        
        # Находим платные каналы
        paid_channels = [name for name, config in self.config['channels'].items() if config['is_paid']]
        
        if not paid_channels:
            print("  ⚠️  Нет платных каналов для анализа зависимостей")
            return
        
        print(f"  📊 АНАЛИЗ ВЛИЯНИЯ БЮДЖЕТА НА ТРАФИК:")
        
        for channel_name in paid_channels:
            channel_config = self.config['channels'][channel_name]
            budget_field = channel_config.get('budget_field')
            
            if not budget_field or budget_field not in self.df.columns:
                print(f"    ⚠️  Поле бюджета '{budget_field}' не найдено для канала {channel_name}")
                continue
            
            # Получаем данные канала
            channel_data = self.df[self.df['channel_type'] == channel_name]
            
            if len(channel_data) < 10:
                print(f"    ⚠️  Недостаточно данных для канала {channel_name}")
                continue
            
            print(f"    📊 Канал {channel_name}:")
            
            # Анализируем влияние бюджета на трафик
            traffic_fields = ['first_traffic', 'repeat_traffic', 'traffic_total']
            
            for traffic_field in traffic_fields:
                if traffic_field in channel_data.columns:
                    correlation = channel_data[budget_field].corr(channel_data[traffic_field])
                    print(f"      {traffic_field}: корреляция с {budget_field} = {correlation:.3f}")
                    
                    # Сохраняем зависимости
                    if channel_name not in self.channel_dependencies:
                        self.channel_dependencies[channel_name] = {}
                    
                    self.channel_dependencies[channel_name][traffic_field] = {
                        'budget_correlation': correlation,
                        'data_points': len(channel_data)
                    }
        
        # Анализ кросс-канального влияния
        self._analyze_cross_channel_influence()
    
    def _analyze_cross_channel_influence(self):
        """Анализ кросс-канального влияния"""
        print(f"\n  📊 КРОСС-КАНАЛЬНОЕ ВЛИЯНИЕ:")
        
        # Находим платные и органические каналы
        paid_channels = [name for name, config in self.config['channels'].items() if config['is_paid']]
        organic_channels = [name for name, config in self.config['channels'].items() if not config['is_paid']]
        
        if not paid_channels or not organic_channels:
            print("    ⚠️  Нужны и платные, и органические каналы для анализа кросс-канального влияния")
            return
        
        # Агрегируем данные по времени
        time_fields = list(self.config['time_fields'].values())
        
        # Данные платных каналов
        paid_data = self.df[self.df['channel_type'].isin(paid_channels)]
        if len(paid_data) > 10:
            paid_agg = paid_data.groupby(time_fields).agg({
                metric: 'sum' for metric in self.config['metrics'].values() 
                if metric in paid_data.columns
            }).reset_index()
            
            # Данные органических каналов
            organic_data = self.df[self.df['channel_type'].isin(organic_channels)]
            if len(organic_data) > 10:
                organic_agg = organic_data.groupby(time_fields).agg({
                    metric: 'sum' for metric in self.config['metrics'].values() 
                    if metric in organic_data.columns
                }).reset_index()
                
                # Объединяем данные
                merged_data = pd.merge(paid_agg, organic_agg, on=time_fields, suffixes=('_paid', '_organic'))
                
                if len(merged_data) > 5:
                    # Анализируем корреляции
                    for paid_channel in paid_channels:
                        paid_config = self.config['channels'][paid_channel]
                        budget_field = paid_config.get('budget_field')
                        
                        if budget_field and f"{budget_field}_paid" in merged_data.columns:
                            print(f"    📊 Влияние {budget_field} (платные) на органические каналы:")
                            
                            for metric in ['first_traffic', 'repeat_traffic', 'traffic_total']:
                                if f"{metric}_organic" in merged_data.columns:
                                    correlation = merged_data[f"{budget_field}_paid"].corr(merged_data[f"{metric}_organic"])
                                    print(f"      {metric}: корреляция = {correlation:.3f}")
    
    def build_channel_models(self):
        """Построение моделей для разных каналов"""
        print(f"\n🤖 ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ КАНАЛОВ:")
        
        # Подготавливаем данные для обучения
        train_data = self._prepare_training_data()
        
        # Строим модели для каждого канала
        for channel_name, channel_config in self.config['channels'].items():
            print(f"\n  📊 Канал: {channel_name}")
            
            # Фильтруем данные по каналу
            channel_data = train_data[train_data['channel_type'] == channel_name].copy()
            
            if len(channel_data) == 0:
                print(f"    ⚠️  Нет данных для канала {channel_name}")
                continue
            
            print(f"    📈 Данных для обучения: {len(channel_data)} записей")
            
            # Строим модели для метрик канала
            channel_models = {}
            
            for metric_name, metric_field in self.config['metrics'].items():
                if metric_field in channel_data.columns:
                    print(f"    🔧 Обучение модели для {metric_name} ({metric_field})")
                    
                    # Подготавливаем признаки
                    features = self._prepare_features(channel_data)
                    target = channel_data[metric_field]
                    
                    # Обучаем модель
                    model = self._train_model(features, target, metric_name)
                    
                    if model:
                        channel_models[metric_name] = model
                        print(f"      ✅ Модель обучена: R² = {model['r2']:.3f}")
            
            self.models[channel_name] = channel_models
        
        # Строим модели влияния бюджета
        self._build_budget_impact_models()
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Подготовка данных для обучения"""
        # Используем данные до августа 2025 для обучения
        time_fields = list(self.config['time_fields'].values())
        if 'year' in time_fields and 'month' in time_fields:
            train_mask = (
                (self.df['year'] < 2025) | 
                ((self.df['year'] == 2025) & (self.df['month'] <= 8))
            )
            return self.df[train_mask].copy()
        else:
            # Если временные поля не настроены, используем все данные
            return self.df.copy()
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
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
    
    def _train_model(self, features: pd.DataFrame, target: pd.Series, metric_name: str) -> Optional[Dict]:
        """Обучение модели"""
        try:
            # Выбираем числовые признаки
            numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
            X = features[numeric_features].fillna(0)
            y = target.fillna(0)
            
            if len(X) == 0 or len(y) == 0:
                return None
            
            # Масштабирование
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Обучение Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Предсказание
            y_pred = model.predict(X_scaled)
            
            # Метрики
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
            print(f"      ❌ Ошибка в обучении модели: {str(e)}")
            return None
    
    def _build_budget_impact_models(self):
        """Построение моделей влияния бюджета на трафик"""
        print(f"\n💰 ПОСТРОЕНИЕ МОДЕЛЕЙ ВЛИЯНИЯ БЮДЖЕТА:")
        
        # Строим модели для платных каналов
        paid_channels = [name for name, config in self.config['channels'].items() if config['is_paid']]
        
        for channel_name in paid_channels:
            channel_config = self.config['channels'][channel_name]
            budget_field = channel_config.get('budget_field')
            
            if not budget_field or budget_field not in self.df.columns:
                continue
            
            channel_data = self.df[self.df['channel_type'] == channel_name].copy()
            
            if len(channel_data) > 20:
                print(f"  📊 Канал {channel_name}: {len(channel_data)} записей")
                
                # Модель влияния бюджета на трафик
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
                                'intercept': model.intercept_,
                                'channel': channel_name,
                                'budget_field': budget_field,
                                'traffic_field': traffic_field
                            }
                            
                            print(f"    ✅ {budget_field} -> {traffic_field}: R² = {r2:.3f}, коэффициент = {model.coef_[0]:.6f}")
    
    def create_forecast(self, forecast_periods: int = 16):
        """Создание прогноза"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА НА {forecast_periods} ПЕРИОДОВ:")
        
        if not self.models:
            print("❌ Нет обученных моделей")
            return
        
        # Создаем прогноз для каждого канала
        for channel_name, channel_models in self.models.items():
            print(f"\n  📊 Прогноз для канала: {channel_name}")
            
            channel_forecast = {}
            
            for metric_name, model_info in channel_models.items():
                print(f"    🔮 Прогноз {metric_name}")
                
                # Создаем будущие периоды
                future_data = self._create_future_periods(forecast_periods)
                
                # Подготавливаем признаки для прогноза
                future_features = self._prepare_features(future_data)
                
                # Прогноз
                forecast_values = self._make_prediction(model_info, future_features)
                
                channel_forecast[metric_name] = {
                    'values': forecast_values,
                    'periods': future_data[['year', 'month']].to_dict('records')
                }
                
                total_forecast = np.sum(forecast_values)
                print(f"      Общий прогноз: {total_forecast:,.0f}")
            
            self.forecasts[channel_name] = channel_forecast
    
    def _create_future_periods(self, periods: int) -> pd.DataFrame:
        """Создание будущих периодов"""
        # Начинаем с сентября 2025
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
    
    def _make_prediction(self, model_info: Dict, features: pd.DataFrame) -> np.ndarray:
        """Создание прогноза"""
        try:
            # Выбираем нужные признаки
            model_features = model_info['features']
            X = features[model_features].fillna(0)
            
            # Масштабирование
            X_scaled = model_info['scaler'].transform(X)
            
            # Прогноз
            predictions = model_info['model'].predict(X_scaled)
            
            # Убираем отрицательные значения
            return np.maximum(predictions, 0)
            
        except Exception as e:
            print(f"      ❌ Ошибка в прогнозе: {str(e)}")
            return np.zeros(len(features))
    
    def simulate_budget_scenarios(self, budget_scenarios: List[Dict]):
        """Моделирование различных сценариев бюджета"""
        print(f"\n🎯 МОДЕЛИРОВАНИЕ СЦЕНАРИЕВ БЮДЖЕТА:")
        
        if not self.forecasts:
            print("❌ Нет прогнозов для моделирования")
            return
        
        for i, scenario in enumerate(budget_scenarios):
            print(f"\n  📊 Сценарий {i+1}: {scenario.get('name', f'Сценарий {i+1}')}")
            
            # Применяем изменения бюджета
            modified_forecast = self._apply_budget_changes(scenario)
            
            # Рассчитываем влияние на другие метрики
            impact_analysis = self._analyze_budget_impact(scenario, modified_forecast)
            
            self.simulation_results[f"scenario_{i+1}"] = {
                'scenario': scenario,
                'modified_forecast': modified_forecast,
                'impact_analysis': impact_analysis
            }
            
            print(f"    💰 Изменение бюджета: {scenario.get('budget_change', 0):+.1f}%")
            print(f"    📈 Влияние на выручку: {impact_analysis.get('revenue_impact', 0):+.1f}%")
    
    def _apply_budget_changes(self, scenario: Dict) -> Dict:
        """Применение изменений бюджета к прогнозу"""
        modified_forecast = {}
        
        for channel_name, channel_forecast in self.forecasts.items():
            modified_channel = {}
            
            for metric_name, metric_forecast in channel_forecast.items():
                values = metric_forecast['values'].copy()
                
                # Применяем изменения к полям бюджета
                channel_config = self.config['channels'][channel_name]
                budget_field = channel_config.get('budget_field')
                
                if budget_field and metric_name == budget_field and 'budget_change' in scenario:
                    change_factor = 1 + scenario['budget_change'] / 100
                    values = values * change_factor
                
                # Применяем влияние на трафик через модели влияния бюджета
                if 'budget_change' in scenario:
                    model_key = f'{channel_name}_{budget_field}_to_{metric_name}'
                    if model_key in self.budget_impact_models:
                        model_info = self.budget_impact_models[model_key]
                        
                        # Рассчитываем изменение трафика на основе изменения бюджета
                        budget_change = scenario['budget_change'] / 100
                        traffic_change = budget_change * model_info['coefficient']
                        
                        # Применяем изменение к прогнозу
                        values = values * (1 + traffic_change)
                        
                        print(f"      {metric_name}: изменение на {traffic_change*100:+.1f}% (коэффициент: {model_info['coefficient']:.6f})")
                
                modified_channel[metric_name] = {
                    'values': values,
                    'periods': metric_forecast['periods']
                }
            
            modified_forecast[channel_name] = modified_channel
        
        return modified_forecast
    
    def _analyze_budget_impact(self, scenario: Dict, modified_forecast: Dict) -> Dict:
        """Анализ влияния изменений бюджета"""
        impact_analysis = {}
        
        # Сравниваем с базовым прогнозом
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
    
    def generate_report(self, output_file: str = 'User_Configurable_Marketing_Report.txt'):
        """Генерация отчета"""
        print(f"\n📊 ГЕНЕРАЦИЯ ОТЧЕТА:")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ МАРКЕТИНГА (ПОЛЬЗОВАТЕЛЬСКИ НАСТРАИВАЕМЫЙ)\n")
            f.write("=" * 70 + "\n\n")
            
            # Конфигурация
            f.write("КОНФИГУРАЦИЯ:\n")
            f.write(f"Временные поля: {list(self.config['time_fields'].values())}\n")
            f.write(f"Измерения: {list(self.config['dimensions'].values())}\n")
            f.write(f"Метрики: {list(self.config['metrics'].values())}\n")
            f.write(f"Каналы: {list(self.config['channels'].keys())}\n\n")
            
            # Детали каналов
            f.write("ДЕТАЛИ КАНАЛОВ:\n")
            for channel_name, channel_config in self.config['channels'].items():
                channel_type = "платный" if channel_config['is_paid'] else "органический"
                budget_field = channel_config.get('budget_field', 'нет')
                f.write(f"{channel_name}: {channel_type}, бюджет: {budget_field}\n")
            f.write("\n")
            
            # Зависимости
            if self.channel_dependencies:
                f.write("ЗАВИСИМОСТИ МЕЖДУ КАНАЛАМИ:\n")
                for channel, deps in self.channel_dependencies.items():
                    f.write(f"\nКанал {channel}:\n")
                    if isinstance(deps, dict):
                        for metric, metric_deps in deps.items():
                            if isinstance(metric_deps, dict) and 'budget_correlation' in metric_deps:
                                f.write(f"  {metric}: корреляция с бюджетом = {metric_deps['budget_correlation']:.3f}\n")
                f.write("\n")
            
            # Модели влияния бюджета
            if self.budget_impact_models:
                f.write("МОДЕЛИ ВЛИЯНИЯ БЮДЖЕТА:\n")
                for model_name, model_info in self.budget_impact_models.items():
                    f.write(f"{model_name}: R² = {model_info['r2']:.3f}, коэффициент = {model_info['coefficient']:.6f}\n")
                f.write("\n")
            
            # Прогнозы
            if self.forecasts:
                f.write("ПРОГНОЗЫ:\n")
                for channel_name, channel_forecast in self.forecasts.items():
                    f.write(f"\nКанал: {channel_name}\n")
                    for metric_name, metric_forecast in channel_forecast.items():
                        total = np.sum(metric_forecast['values'])
                        f.write(f"  {metric_name}: {total:,.0f}\n")
                f.write("\n")
            
            # Результаты моделирования
            if self.simulation_results:
                f.write("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:\n")
                for scenario_name, results in self.simulation_results.items():
                    f.write(f"\n{scenario_name}:\n")
                    f.write(f"  Сценарий: {results['scenario']}\n")
                    f.write(f"  Влияние на выручку: {results['impact_analysis'].get('revenue_impact', 0):+.1f}%\n")
        
        print(f"✅ Отчет сохранен в файл: {output_file}")
    
    def save_results(self, output_file: str = 'User_Configurable_Marketing_Results.csv'):
        """Сохранение результатов"""
        if not self.forecasts:
            print("❌ Нет результатов для сохранения")
            return
        
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
        results_df.to_csv(output_file, index=False)
        
        print(f"💾 Результаты сохранены в файл: {output_file}")

def main():
    """Основная функция"""
    print("🚀 ПОЛЬЗОВАТЕЛЬСКИ НАСТРАИВАЕМЫЙ ИНСТРУМЕНТ ПРОГНОЗИРОВАНИЯ МАРКЕТИНГА")
    print("=" * 70)
    
    # Инициализация
    tool = UserConfigurableMarketingTool()
    
    # Загрузка данных
    if not tool.load_data('Marketing Budjet Emulation - raw2.csv'):
        return
    
    # Настройка конфигурации (по умолчанию)
    tool.setup_default_configuration()
    
    # Определение каналов
    tool.identify_channels()
    
    # Анализ зависимостей
    tool.analyze_channel_dependencies()
    
    # Построение моделей каналов
    tool.build_channel_models()
    
    # Создание прогноза
    tool.create_forecast()
    
    # Моделирование сценариев бюджета
    budget_scenarios = [
        {'name': 'Увеличение бюджета на 20%', 'budget_change': 20},
        {'name': 'Уменьшение бюджета на 10%', 'budget_change': -10},
        {'name': 'Увеличение бюджета на 50%', 'budget_change': 50}
    ]
    tool.simulate_budget_scenarios(budget_scenarios)
    
    # Генерация отчета
    tool.generate_report()
    
    # Сохранение результатов
    tool.save_results()
    
    print(f"\n🎉 Пользовательски настраиваемый инструмент прогнозирования завершен!")

if __name__ == "__main__":
    main()
