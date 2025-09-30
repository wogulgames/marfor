#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный инструмент для прогнозирования и бюджетирования маркетинга
Включает зависимости между каналами и моделирование влияния бюджета
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

class UniversalMarketingForecastTool:
    def __init__(self):
        """Инициализация универсального инструмента прогнозирования"""
        self.df = None
        self.config = {}
        self.channel_dependencies = {}
        self.cascade_config = {}
        self.models = {}
        self.forecasts = {}
        self.simulation_results = {}
        
    def setup_configuration(self, config_file: str = None, config_dict: Dict = None):
        """Настройка конфигурации инструмента"""
        print("🔧 НАСТРОЙКА КОНФИГУРАЦИИ ИНСТРУМЕНТА:")
        
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # Конфигурация по умолчанию для текущего датасета
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
                    "paid": {
                        "is_paid": True,
                        "traffic_fields": ["first_traffic", "repeat_traffic"],
                        "budget_field": "ads_cost",
                        "conversion_fields": ["first_transactions", "repeat_transactions"]
                    },
                    "organic": {
                        "is_paid": False,
                        "traffic_fields": ["first_traffic", "repeat_traffic"],
                        "budget_field": None,
                        "conversion_fields": ["first_transactions", "repeat_transactions"]
                    }
                },
                "cascade": {
                    "enabled": True,
                    "levels": ["region", "subdivision", "category"],
                    "outlier_threshold": 2.0,
                    "missing_value_strategy": "cascade_up"
                },
                "dependencies": {
                    "ads_cost_to_traffic": {
                        "enabled": True,
                        "paid_channels_only": True,
                        "cross_channel_influence": 0.1
                    }
                }
            }
        
        print("✅ Конфигурация загружена")
        self._print_config_summary()
    
    def _print_config_summary(self):
        """Вывод сводки конфигурации"""
        print(f"\n📋 СВОДКА КОНФИГУРАЦИИ:")
        print(f"  Временные поля: {list(self.config['time_fields'].values())}")
        print(f"  Измерения: {list(self.config['dimensions'].values())}")
        print(f"  Метрики: {list(self.config['metrics'].values())}")
        print(f"  Каналы: {list(self.config['channels'].keys())}")
        print(f"  Каскад: {'Включен' if self.config['cascade']['enabled'] else 'Отключен'}")
        print(f"  Зависимости: {'Включены' if self.config['dependencies']['ads_cost_to_traffic']['enabled'] else 'Отключены'}")
    
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
        
        # Очистка временных полей
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
    
    def analyze_channel_dependencies(self):
        """Анализ зависимостей между каналами"""
        print(f"\n🔍 АНАЛИЗ ЗАВИСИМОСТЕЙ МЕЖДУ КАНАЛАМИ:")
        
        if not self.config['dependencies']['ads_cost_to_traffic']['enabled']:
            print("  ⚠️  Анализ зависимостей отключен в конфигурации")
            return
        
        # Анализируем влияние ads_cost на трафик
        ads_cost_field = self.config['metrics']['ads_cost']
        traffic_fields = ['first_traffic', 'repeat_traffic']
        
        if ads_cost_field not in self.df.columns:
            print(f"  ❌ Поле {ads_cost_field} не найдено в данных")
            return
        
        print(f"\n  📊 ВЛИЯНИЕ {ads_cost_field} НА ТРАФИК:")
        
        for traffic_field in traffic_fields:
            if traffic_field in self.df.columns:
                # Корреляция
                correlation = self.df[ads_cost_field].corr(self.df[traffic_field])
                print(f"    {traffic_field}: корреляция = {correlation:.3f}")
                
                # Анализ по каналам
                if 'is_paid' in self.df.columns:
                    paid_corr = self.df[self.df['is_paid'] == True][ads_cost_field].corr(
                        self.df[self.df['is_paid'] == True][traffic_field]
                    )
                    organic_corr = self.df[self.df['is_paid'] == False][ads_cost_field].corr(
                        self.df[self.df['is_paid'] == False][traffic_field]
                    )
                    print(f"      Платные каналы: {paid_corr:.3f}")
                    print(f"      Органические каналы: {organic_corr:.3f}")
                
                # Сохраняем зависимости
                self.channel_dependencies[traffic_field] = {
                    'ads_cost_correlation': correlation,
                    'paid_correlation': paid_corr if 'is_paid' in self.df.columns else None,
                    'organic_correlation': organic_corr if 'is_paid' in self.df.columns else None
                }
    
    def build_cascade_forecast(self):
        """Построение каскадного прогноза"""
        print(f"\n🏗️ ПОСТРОЕНИЕ КАСКАДНОГО ПРОГНОЗА:")
        
        if not self.config['cascade']['enabled']:
            print("  ⚠️  Каскадный прогноз отключен в конфигурации")
            return
        
        # Агрегируем данные по уровням каскада
        cascade_levels = self.config['cascade']['levels']
        time_fields = list(self.config['time_fields'].values())
        
        print(f"  📊 АГРЕГАЦИЯ ПО УРОВНЯМ КАСКАДА:")
        
        for level in cascade_levels:
            if level in self.config['dimensions']:
                dimension_field = self.config['dimensions'][level]
                if dimension_field in self.df.columns:
                    print(f"    Уровень: {level} ({dimension_field})")
                    
                    # Агрегируем данные
                    agg_data = self.df.groupby(time_fields + [dimension_field]).agg({
                        metric: 'sum' for metric in self.config['metrics'].values() 
                        if metric in self.df.columns
                    }).reset_index()
                    
                    print(f"      Агрегировано {len(agg_data)} записей")
                    
                    # Анализируем стабильность на этом уровне
                    self._analyze_level_stability(agg_data, level, dimension_field)
    
    def _analyze_level_stability(self, data: pd.DataFrame, level: str, dimension_field: str):
        """Анализ стабильности данных на уровне каскада"""
        print(f"      🔍 Анализ стабильности уровня {level}:")
        
        revenue_field = self.config['metrics']['revenue']
        if revenue_field not in data.columns:
            return
        
        for dimension_value in data[dimension_field].unique():
            dimension_data = data[data[dimension_field] == dimension_value].copy()
            
            if len(dimension_data) > 6:  # Минимум 6 месяцев данных
                revenue_values = dimension_data[revenue_field].values
                
                # Коэффициент вариации
                cv = np.std(revenue_values) / np.mean(revenue_values) if np.mean(revenue_values) > 0 else 0
                
                # Тренд
                time_index = np.arange(len(revenue_values))
                trend_slope = np.polyfit(time_index, revenue_values, 1)[0]
                trend_strength = abs(trend_slope) / np.mean(revenue_values) if np.mean(revenue_values) > 0 else 0
                
                # Оценка стабильности
                stability_score = (1 - min(cv, 1)) * 0.5 + (1 - min(trend_strength, 1)) * 0.5
                
                print(f"        {dimension_value}: стабильность = {stability_score:.3f}")
    
    def build_channel_models(self):
        """Построение моделей для разных каналов"""
        print(f"\n🤖 ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ КАНАЛОВ:")
        
        # Подготавливаем данные для обучения
        train_data = self._prepare_training_data()
        
        # Строим модели для каждого канала
        for channel_name, channel_config in self.config['channels'].items():
            print(f"\n  📊 Канал: {channel_name}")
            
            # Фильтруем данные по каналу
            if 'is_paid' in self.df.columns:
                if channel_config['is_paid']:
                    channel_data = train_data[train_data['is_paid'] == True].copy()
                else:
                    channel_data = train_data[train_data['is_paid'] == False].copy()
            else:
                channel_data = train_data.copy()
            
            if len(channel_data) == 0:
                print(f"    ⚠️  Нет данных для канала {channel_name}")
                continue
            
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
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Подготовка данных для обучения"""
        # Используем данные до августа 2025 для обучения
        train_mask = (
            (self.df['year'] < 2025) | 
            ((self.df['year'] == 2025) & (self.df['month'] <= 8))
        )
        return self.df[train_mask].copy()
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        features_df = data.copy()
        
        # Временные признаки
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
                
                # Применяем изменения к ads_cost
                if metric_name == 'ads_cost' and 'budget_change' in scenario:
                    change_factor = 1 + scenario['budget_change'] / 100
                    values = values * change_factor
                
                # Применяем влияние на трафик
                if metric_name in ['first_traffic', 'repeat_traffic'] and 'budget_change' in scenario:
                    # Используем зависимости между каналами
                    if metric_name in self.channel_dependencies:
                        correlation = self.channel_dependencies[metric_name]['ads_cost_correlation']
                        if not np.isnan(correlation):
                            # Применяем влияние с учетом корреляции
                            influence_factor = 1 + (scenario['budget_change'] / 100) * abs(correlation) * 0.5
                            values = values * influence_factor
                
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
    
    def generate_report(self, output_file: str = 'Marketing_Forecast_Report.txt'):
        """Генерация отчета"""
        print(f"\n📊 ГЕНЕРАЦИЯ ОТЧЕТА:")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ МАРКЕТИНГА\n")
            f.write("=" * 50 + "\n\n")
            
            # Конфигурация
            f.write("КОНФИГУРАЦИЯ:\n")
            f.write(f"Временные поля: {list(self.config['time_fields'].values())}\n")
            f.write(f"Измерения: {list(self.config['dimensions'].values())}\n")
            f.write(f"Метрики: {list(self.config['metrics'].values())}\n")
            f.write(f"Каналы: {list(self.config['channels'].keys())}\n\n")
            
            # Зависимости
            if self.channel_dependencies:
                f.write("ЗАВИСИМОСТИ МЕЖДУ КАНАЛАМИ:\n")
                for metric, deps in self.channel_dependencies.items():
                    f.write(f"{metric}: корреляция с ads_cost = {deps['ads_cost_correlation']:.3f}\n")
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
    
    def save_results(self, output_file: str = 'Marketing_Forecast_Results.csv'):
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
    print("🚀 УНИВЕРСАЛЬНЫЙ ИНСТРУМЕНТ ПРОГНОЗИРОВАНИЯ МАРКЕТИНГА")
    print("=" * 60)
    
    # Инициализация
    tool = UniversalMarketingForecastTool()
    
    # Настройка конфигурации
    tool.setup_configuration()
    
    # Загрузка данных
    if not tool.load_data('Marketing Budjet Emulation - raw2.csv'):
        return
    
    # Анализ зависимостей
    tool.analyze_channel_dependencies()
    
    # Построение каскадного прогноза
    tool.build_cascade_forecast()
    
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
    
    print(f"\n🎉 Универсальный инструмент прогнозирования завершен!")

if __name__ == "__main__":
    main()
