#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Каскадная модель прогнозирования с агрегацией по уровням
1. Верхнеуровневые тренды по регионам
2. Использование трендов для улучшения низкоуровневых данных
3. Построение финального прогноза
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CascadedForecaster:
    def __init__(self, csv_file=None):
        """Инициализация каскадного прогнозировщика"""
        self.csv_file = csv_file
        self.df = None
        self.aggregated_df = None
        self.region_trends = {}
        self.region_models = {}
        self.final_forecast = None
        
    def load_and_analyze_data(self, csv_file=None):
        """Загрузка и анализ данных"""
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
        
        return self.df
    
    def clean_data(self):
        """Очистка и подготовка данных"""
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
        
        # Анализ временного диапазона
        print(f"\n📅 ВРЕМЕННОЙ ДИАПАЗОН:")
        print(f"  С {self.df['year'].min()}.{self.df['month'].min():02d} по {self.df['year'].max()}.{self.df['month'].max():02d}")
        
        # Анализ измерений
        print(f"\n📊 АНАЛИЗ ИЗМЕРЕНИЙ:")
        if 'region_to' in self.df.columns:
            regions = self.df['region_to'].value_counts()
            print(f"  Регионы: {len(regions)} ({list(regions.index)})")
        
        if 'subdivision' in self.df.columns:
            subdivisions = self.df['subdivision'].value_counts()
            print(f"  Подразделения: {len(subdivisions)} ({list(subdivisions.index)})")
        
        if 'category' in self.df.columns:
            categories = self.df['category'].value_counts()
            print(f"  Категории: {len(categories)} ({list(categories.index)})")
    
    def aggregate_by_region(self):
        """Агрегация данных по регионам для построения верхнеуровневых трендов"""
        print(f"\n📈 АГРЕГАЦИЯ ДАННЫХ ПО РЕГИОНАМ:")
        
        # Агрегируем данные по регионам и месяцам
        if 'region_to' in self.df.columns and 'revenue_total' in self.df.columns:
            self.aggregated_df = self.df.groupby(['year', 'month', 'region_to'])['revenue_total'].agg([
                'sum', 'mean', 'count', 'std'
            ]).reset_index()
            
            # Переименовываем колонки
            self.aggregated_df.columns = ['year', 'month', 'region_to', 'total_revenue', 'avg_revenue', 'count', 'std_revenue']
            
            print(f"  Агрегировано {len(self.aggregated_df)} записей по регионам")
            
            # Анализ агрегированных данных
            print(f"\n📊 АНАЛИЗ АГРЕГИРОВАННЫХ ДАННЫХ:")
            print(f"  Общая выручка по регионам:")
            region_totals = self.aggregated_df.groupby('region_to')['total_revenue'].sum().sort_values(ascending=False)
            for region, total in region_totals.items():
                print(f"    {region}: {total:,.0f} ₽")
            
            # Анализ временных трендов
            print(f"\n📅 ВРЕМЕННЫЕ ТРЕНДЫ ПО РЕГИОНАМ:")
            for region in self.aggregated_df['region_to'].unique():
                region_data = self.aggregated_df[self.aggregated_df['region_to'] == region]
                if len(region_data) > 1:
                    # Создаем временной индекс
                    region_data = region_data.sort_values(['year', 'month'])
                    region_data['time_index'] = (region_data['year'] - region_data['year'].min()) * 12 + (region_data['month'] - 1)
                    
                    # Простой линейный тренд
                    if len(region_data) > 2:
                        X = region_data[['time_index']]
                        y = region_data['total_revenue']
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        trend_slope = model.coef_[0]
                        trend_direction = "рост" if trend_slope > 0 else "падение" if trend_slope < 0 else "стабильно"
                        
                        print(f"    {region}: {trend_direction} ({trend_slope:,.0f} ₽/месяц)")
                        
                        # Сохраняем тренд
                        self.region_trends[region] = {
                            'slope': trend_slope,
                            'intercept': model.intercept_,
                            'data': region_data,
                            'model': model
                        }
        else:
            print("❌ Не найдены необходимые колонки для агрегации")
    
    def analyze_seasonality(self):
        """Анализ сезонности по регионам"""
        print(f"\n🌊 АНАЛИЗ СЕЗОННОСТИ ПО РЕГИОНАМ:")
        
        if self.aggregated_df is not None:
            # Анализ по месяцам
            monthly_avg = self.aggregated_df.groupby('month')['total_revenue'].mean()
            print(f"  Средняя выручка по месяцам:")
            for month, revenue in monthly_avg.items():
                print(f"    {month:2d} месяц: {revenue:,.0f} ₽")
            
            # Анализ по кварталам
            self.aggregated_df['quarter'] = ((self.aggregated_df['month'] - 1) // 3) + 1
            quarterly_avg = self.aggregated_df.groupby('quarter')['total_revenue'].mean()
            print(f"\n  Средняя выручка по кварталам:")
            for quarter, revenue in quarterly_avg.items():
                print(f"    Q{quarter}: {revenue:,.0f} ₽")
            
            # Анализ сезонности по регионам
            print(f"\n🌍 СЕЗОННОСТЬ ПО РЕГИОНАМ:")
            for region in self.aggregated_df['region_to'].unique():
                region_data = self.aggregated_df[self.aggregated_df['region_to'] == region]
                if len(region_data) > 6:  # Минимум 6 месяцев данных
                    # Анализ по месяцам
                    monthly_revenue = region_data.groupby('month')['total_revenue'].mean()
                    peak_month = monthly_revenue.idxmax()
                    low_month = monthly_revenue.idxmin()
                    peak_value = monthly_revenue.max()
                    low_value = monthly_revenue.min()
                    
                    seasonality_ratio = peak_value / low_value if low_value > 0 else 0
                    
                    print(f"    {region}:")
                    print(f"      Пик: {peak_month} месяц ({peak_value:,.0f} ₽)")
                    print(f"      Минимум: {low_month} месяц ({low_value:,.0f} ₽)")
                    print(f"      Сезонность: {seasonality_ratio:.2f}x")
    
    def build_region_models(self):
        """Построение моделей для каждого региона"""
        print(f"\n🤖 ПОСТРОЕНИЕ МОДЕЛЕЙ ДЛЯ РЕГИОНОВ:")
        
        if self.aggregated_df is not None:
            for region in self.aggregated_df['region_to'].unique():
                region_data = self.aggregated_df[self.aggregated_df['region_to'] == region].copy()
                
                if len(region_data) > 6:  # Минимум 6 месяцев данных
                    print(f"\n  🔧 Обучение модели для {region}:")
                    
                    # Подготавливаем данные
                    region_data = region_data.sort_values(['year', 'month'])
                    region_data['time_index'] = (region_data['year'] - region_data['year'].min()) * 12 + (region_data['month'] - 1)
                    
                    # Сезонные признаки
                    region_data['month_sin'] = np.sin(2 * np.pi * region_data['month'] / 12)
                    region_data['month_cos'] = np.cos(2 * np.pi * region_data['month'] / 12)
                    
                    # Квартальные признаки
                    for q in range(1, 5):
                        region_data[f'q{q}'] = (region_data['quarter'] == q).astype(int)
                    
                    # Праздничные периоды
                    region_data['holiday_period'] = (
                        (region_data['month'] == 12) |  # Декабрь
                        (region_data['month'] == 1) |   # Январь
                        (region_data['month'] == 2) |   # Февраль
                        (region_data['month'] == 3) |   # Март
                        (region_data['month'] == 5)     # Май
                    ).astype(int)
                    
                    # Подготавливаем X и y
                    features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
                    X = region_data[features].fillna(0)
                    y = region_data['total_revenue']
                    
                    # Обучение модели
                    model = Ridge(alpha=1.0)
                    model.fit(X, y)
                    
                    # Предсказание
                    y_pred = model.predict(X)
                    
                    # Метрики
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    print(f"    R²: {r2:.3f}")
                    print(f"    MAE: {mae:,.0f}")
                    
                    # Сохраняем модель
                    self.region_models[region] = {
                        'model': model,
                        'features': features,
                        'r2': r2,
                        'mae': mae,
                        'data': region_data
                    }
                    
                    if r2 > 0.3:
                        print(f"    ✅ Модель хорошего качества")
                    elif r2 > 0.1:
                        print(f"    ⚠️  Модель удовлетворительного качества")
                    else:
                        print(f"    ❌ Модель слабого качества")
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза на основе региональных моделей"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА НА {forecast_periods} ПЕРИОДОВ:")
        
        if not self.region_models:
            print("❌ Нет обученных моделей для прогнозирования")
            return None
        
        # Определяем последний временной индекс
        last_year = self.aggregated_df['year'].max()
        last_month = self.aggregated_df['month'].max()
        
        print(f"  Последний период: {last_year}.{last_month:02d}")
        
        # Создаем прогноз для каждого региона
        forecast_data = []
        
        for region, model_info in self.region_models.items():
            print(f"\n  📊 Прогноз для {region}:")
            
            # Получаем последние данные региона
            region_data = model_info['data']
            last_time_index = region_data['time_index'].max()
            
            # Создаем периоды для прогноза
            for i in range(1, forecast_periods + 1):
                period_data = {
                    'year': last_year + (i // 12),
                    'month': ((last_month + i - 1) % 12) + 1,
                    'region_to': region,
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
                features = model_info['features']
                X_forecast = np.array([period_data[feature] for feature in features]).reshape(1, -1)
                forecast_value = model_info['model'].predict(X_forecast)[0]
                
                period_data['forecast_revenue'] = max(0, forecast_value)  # Не допускаем отрицательные значения
                
                forecast_data.append(period_data)
                
                print(f"    {period_data['year']}.{period_data['month']:02d}: {forecast_value:,.0f} ₽")
        
        # Создаем DataFrame с прогнозом
        self.final_forecast = pd.DataFrame(forecast_data)
        
        # Анализ прогноза
        total_forecast = self.final_forecast['forecast_revenue'].sum()
        print(f"\n  📊 ОБЩИЙ ПРОГНОЗ: {total_forecast:,.0f} ₽")
        
        # Анализ по регионам
        print(f"\n  🌍 ПРОГНОЗ ПО РЕГИОНАМ:")
        region_forecasts = self.final_forecast.groupby('region_to')['forecast_revenue'].sum().sort_values(ascending=False)
        for region, forecast in region_forecasts.items():
            print(f"    {region}: {forecast:,.0f} ₽")
        
        return self.final_forecast
    
    def save_forecast(self, output_file='Cascaded_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.final_forecast is None:
            print("❌ Прогноз не создан")
            return None
        
        # Сохраняем прогноз
        self.final_forecast.to_csv(output_file, index=False)
        print(f"\n💾 Прогноз сохранен в файл: {output_file}")
        
        return self.final_forecast
    
    def plot_analysis(self, save_plot=True):
        """Визуализация анализа"""
        if self.aggregated_df is None:
            print("❌ Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График 1: Тренды по регионам
        ax1 = axes[0, 0]
        for region in self.aggregated_df['region_to'].unique():
            region_data = self.aggregated_df[self.aggregated_df['region_to'] == region]
            region_data = region_data.sort_values(['year', 'month'])
            region_data['period'] = region_data['year'] + region_data['month'] / 12
            ax1.plot(region_data['period'], region_data['total_revenue'], 
                    label=region, marker='o', linewidth=2)
        
        ax1.set_title('Тренды по регионам')
        ax1.set_xlabel('Период')
        ax1.set_ylabel('Выручка')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Сезонность
        ax2 = axes[0, 1]
        monthly_avg = self.aggregated_df.groupby('month')['total_revenue'].mean()
        ax2.bar(monthly_avg.index, monthly_avg.values, alpha=0.7)
        ax2.set_title('Сезонность (средняя выручка по месяцам)')
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Выручка')
        ax2.grid(True, alpha=0.3)
        
        # График 3: Распределение по регионам
        ax3 = axes[1, 0]
        region_totals = self.aggregated_df.groupby('region_to')['total_revenue'].sum().sort_values(ascending=False)
        ax3.bar(range(len(region_totals)), region_totals.values, alpha=0.7)
        ax3.set_title('Общая выручка по регионам')
        ax3.set_xlabel('Регионы')
        ax3.set_ylabel('Выручка')
        ax3.set_xticks(range(len(region_totals)))
        ax3.set_xticklabels(region_totals.index, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # График 4: Качество моделей
        ax4 = axes[1, 1]
        if self.region_models:
            regions = list(self.region_models.keys())
            r2_scores = [self.region_models[region]['r2'] for region in regions]
            ax4.bar(range(len(regions)), r2_scores, alpha=0.7)
            ax4.set_title('Качество моделей по регионам (R²)')
            ax4.set_xlabel('Регионы')
            ax4.set_ylabel('R²')
            ax4.set_xticks(range(len(regions)))
            ax4.set_xticklabels(regions, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('cascaded_forecast_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 График сохранен как 'cascaded_forecast_analysis.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("🎯 КАСКАДНАЯ МОДЕЛЬ ПРОГНОЗИРОВАНИЯ")
    print("="*60)
    print("1. Агрегация по регионам")
    print("2. Построение верхнеуровневых трендов")
    print("3. Создание прогноза")
    print("="*60)
    
    # Инициализация
    forecaster = CascadedForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и анализ данных
    forecaster.load_and_analyze_data()
    
    # Очистка данных
    forecaster.clean_data()
    
    # Агрегация по регионам
    forecaster.aggregate_by_region()
    
    # Анализ сезонности
    forecaster.analyze_seasonality()
    
    # Построение моделей для регионов
    forecaster.build_region_models()
    
    # Создание прогноза
    forecaster.create_forecast(forecast_periods=4)
    
    # Сохранение результатов
    forecaster.save_forecast()
    
    # Визуализация
    forecaster.plot_analysis()
    
    print(f"\n🎉 Программа завершена успешно!")

if __name__ == "__main__":
    main()
