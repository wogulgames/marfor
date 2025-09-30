#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа прогнозирования с обучением на исторических данных до августа 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HistoricalTrendForecaster:
    def __init__(self, csv_file=None):
        """Инициализация прогнозировщика с историческими данными"""
        self.csv_file = csv_file
        self.df = None
        self.train_df = None
        self.test_df = None
        self.forecast_df = None
        self.model = None
        self.scaler = None
        
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
        
        # Анализ данных по периодам
        if 'revenue_total' in self.df.columns:
            print(f"\n📈 АНАЛИЗ ДАННЫХ ПО ПЕРИОДАМ:")
            period_stats = self.df.groupby(['year', 'month'])['revenue_total'].agg(['count', 'sum', 'mean']).round(0)
            print(f"  Всего периодов: {len(period_stats)}")
            print(f"  Средняя выручка за период: {period_stats['sum'].mean():,.0f} ₽")
            print(f"  Максимальная выручка за период: {period_stats['sum'].max():,.0f} ₽")
    
    def split_historical_data(self, train_end_year=2025, train_end_month=8):
        """Разделение данных на обучающую и тестовую выборки"""
        print(f"\n✂️ РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"  Обучающая выборка: до {train_end_year}.{train_end_month:02d}")
        print(f"  Тестовая выборка: с {train_end_year}.{train_end_month+1:02d}")
        
        # Обучающая выборка (до августа 2025)
        train_mask = (
            (self.df['year'] < train_end_year) | 
            ((self.df['year'] == train_end_year) & (self.df['month'] <= train_end_month))
        )
        self.train_df = self.df[train_mask].copy()
        
        # Тестовая выборка (с сентября 2025)
        test_mask = (
            (self.df['year'] > train_end_year) | 
            ((self.df['year'] == train_end_year) & (self.df['month'] > train_end_month))
        )
        self.test_df = self.df[test_mask].copy()
        
        print(f"  Обучающая выборка: {len(self.train_df)} записей")
        print(f"  Тестовая выборка: {len(self.test_df)} записей")
        
        # Анализ качества разделения
        if 'revenue_total' in self.train_df.columns and 'revenue_total' in self.test_df.columns:
            train_revenue = self.train_df['revenue_total'].sum()
            test_revenue = self.test_df['revenue_total'].sum()
            print(f"  Выручка в обучающей выборке: {train_revenue:,.0f} ₽")
            print(f"  Выручка в тестовой выборке: {test_revenue:,.0f} ₽")
    
    def prepare_time_features(self, df):
        """Подготовка временных признаков"""
        df = df.copy()
        
        # Создаем временной индекс
        df['time_index'] = (df['year'] - df['year'].min()) * 12 + (df['month'] - 1)
        
        # Сезонные признаки
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Квартальные dummy переменные
            df['quarter'] = ((df['month'] - 1) // 3) + 1
            for q in range(1, 5):
                df[f'q{q}'] = (df['quarter'] == q).astype(int)
        
        # Праздничные периоды
        if 'month' in df.columns:
            df['holiday_period'] = (
                (df['month'] == 12) |  # Декабрь
                (df['month'] == 1) |   # Январь
                (df['month'] == 2) |   # Февраль
                (df['month'] == 3) |   # Март
                (df['month'] == 5)     # Май
            ).astype(int)
        
        # Полиномиальные признаки времени
        df['time_squared'] = df['time_index'] ** 2
        df['time_cubed'] = df['time_index'] ** 3
        
        return df
    
    def train_models(self):
        """Обучение различных моделей"""
        print(f"\n🤖 ОБУЧЕНИЕ МОДЕЛЕЙ:")
        
        # Подготавливаем обучающие данные
        train_data = self.prepare_time_features(self.train_df)
        train_data = train_data[train_data['revenue_total'] > 0].copy()
        
        if len(train_data) < 30:
            print(f"❌ Недостаточно данных для обучения ({len(train_data)} записей)")
            return None
        
        print(f"  Данных для обучения: {len(train_data)} записей")
        
        # Базовые временные признаки
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 
                   'holiday_period', 'time_squared', 'time_cubed']
        
        # Подготавливаем X и y
        X_train = train_data[features].fillna(0)
        y_train = train_data['revenue_total']
        
        print(f"  Используемые признаки: {features}")
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Обучение различных моделей
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        model_results = {}
        
        for model_name, model in models.items():
            try:
                # Обучение
                model.fit(X_train_scaled, y_train)
                
                # Предсказание на обучающей выборке
                y_pred_train = model.predict(X_train_scaled)
                
                # Метрики на обучающей выборке
                train_r2 = r2_score(y_train, y_pred_train)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                
                # Валидация на тестовой выборке
                if len(self.test_df) > 0:
                    test_data = self.prepare_time_features(self.test_df)
                    test_data = test_data[test_data['revenue_total'] > 0].copy()
                    
                    if len(test_data) > 0:
                        X_test = test_data[features].fillna(0)
                        y_test = test_data['revenue_total']
                        X_test_scaled = self.scaler.transform(X_test)
                        
                        y_pred_test = model.predict(X_test_scaled)
                        
                        test_r2 = r2_score(y_test, y_pred_test)
                        test_mae = mean_absolute_error(y_test, y_pred_test)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
                        model_results[model_name] = {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse
                        }
                        
                        print(f"\n  {model_name}:")
                        print(f"    Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
                        print(f"    Train MAE: {train_mae:,.0f}, Test MAE: {test_mae:,.0f}")
                        print(f"    Train RMSE: {train_rmse:,.0f}, Test RMSE: {test_rmse:,.0f}")
                        
                        # Выбираем модель с лучшим R² на тестовой выборке
                        if test_r2 > best_score:
                            best_score = test_r2
                            best_model = model
                            best_model_name = model_name
                    else:
                        print(f"  {model_name}: нет тестовых данных для валидации")
                else:
                    print(f"  {model_name}: нет тестовой выборки")
                    
            except Exception as e:
                print(f"  Ошибка при обучении {model_name}: {e}")
                continue
        
        if best_model is not None:
            self.model = best_model
            print(f"\n  🏆 Лучшая модель: {best_model_name} (Test R² = {best_score:.3f})")
            
            # Показываем коэффициенты для линейных моделей
            if hasattr(best_model, 'coef_'):
                print(f"\n  📊 КОЭФФИЦИЕНТЫ МОДЕЛИ:")
                for feature, coef in zip(features, best_model.coef_):
                    print(f"    {feature}: {coef:.2f}")
                print(f"    Intercept: {best_model.intercept_:.2f}")
            
            return {
                'model_name': best_model_name,
                'features': features,
                'test_r2': best_score,
                'train_size': len(train_data),
                'model_results': model_results
            }
        else:
            print(f"❌ Не удалось обучить ни одну модель")
            return None
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза"""
        print(f"\n🔮 СОЗДАНИЕ ПРОГНОЗА НА {forecast_periods} ПЕРИОДОВ:")
        
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train_models()")
        
        # Определяем последний временной индекс
        last_time_index = self.train_df['time_index'].max() if 'time_index' in self.train_df.columns else 0
        
        # Получаем уникальные комбинации групп
        group_cols = ['region_to', 'subdivision', 'category']
        available_group_cols = [col for col in group_cols if col in self.train_df.columns]
        
        if available_group_cols:
            unique_combinations = self.train_df[available_group_cols].drop_duplicates()
            print(f"  Создание прогноза для {len(unique_combinations)} комбинаций групп")
        else:
            unique_combinations = pd.DataFrame({'dummy': [1]})
            print(f"  Создание прогноза без группировки")
        
        # Создаем периоды для прогноза
        forecast_periods_data = []
        for _, combo in unique_combinations.iterrows():
            for i in range(1, forecast_periods + 1):
                period_data = {
                    'time_index': last_time_index + i,
                    'month_sin': np.sin(2 * np.pi * ((last_time_index + i) % 12) / 12),
                    'month_cos': np.cos(2 * np.pi * ((last_time_index + i) % 12) / 12),
                }
                
                # Добавляем групповые данные
                for col in available_group_cols:
                    period_data[col] = combo[col]
                
                # Добавляем квартальные признаки
                month = ((last_time_index + i) % 12) + 1
                quarter = ((month - 1) // 3) + 1
                for q in range(1, 5):
                    period_data[f'q{q}'] = 1 if quarter == q else 0
                
                # Праздничные периоды
                period_data['holiday_period'] = 1 if month in [12, 1, 2, 3, 5] else 0
                
                # Полиномиальные признаки
                period_data['time_squared'] = period_data['time_index'] ** 2
                period_data['time_cubed'] = period_data['time_index'] ** 3
                
                forecast_periods_data.append(period_data)
        
        self.forecast_df = pd.DataFrame(forecast_periods_data)
        
        # Прогнозируем revenue_total
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 
                   'holiday_period', 'time_squared', 'time_cubed']
        forecast_features = self.forecast_df[features].fillna(0)
        forecast_scaled = self.scaler.transform(forecast_features)
        
        # Прогноз
        predictions = self.model.predict(forecast_scaled)
        
        # Сохраняем прогноз
        self.forecast_df['revenue_total'] = np.maximum(0, predictions)
        
        print(f"  Прогноз создан для {len(forecast_periods_data)} записей")
        print(f"  Общий прогноз revenue_total: {self.forecast_df['revenue_total'].sum():,.0f} ₽")
        
        # Анализ прогноза
        print(f"\n  📊 АНАЛИЗ ПРОГНОЗА:")
        print(f"    Средний прогноз на запись: {self.forecast_df['revenue_total'].mean():,.0f} ₽")
        print(f"    Максимальный прогноз: {self.forecast_df['revenue_total'].max():,.0f} ₽")
        print(f"    Минимальный прогноз: {self.forecast_df['revenue_total'].min():,.0f} ₽")
    
    def save_forecast(self, output_file='Historical_Trend_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.forecast_df is None:
            raise ValueError("Прогноз не создан. Сначала выполните create_forecast()")
        
        # Сохраняем прогноз
        self.forecast_df.to_csv(output_file, index=False)
        print(f"\n💾 Прогноз сохранен в файл: {output_file}")
        
        return self.forecast_df
    
    def plot_forecast_analysis(self, save_plot=True):
        """Визуализация анализа прогноза"""
        if self.forecast_df is None:
            print("Прогноз не создан")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Подграфик 1: Временной ряд
        plt.subplot(2, 2, 1)
        
        # Исторические данные (обучающая выборка)
        train_data = self.prepare_time_features(self.train_df)
        train_data = train_data[train_data['revenue_total'] > 0]
        if len(train_data) > 0:
            plt.plot(train_data['time_index'], train_data['revenue_total'], 
                    'b-', label='Обучающие данные', linewidth=2, alpha=0.7)
        
        # Тестовые данные
        if len(self.test_df) > 0:
            test_data = self.prepare_time_features(self.test_df)
            test_data = test_data[test_data['revenue_total'] > 0]
            if len(test_data) > 0:
                plt.plot(test_data['time_index'], test_data['revenue_total'], 
                        'g-', label='Тестовые данные', linewidth=2, alpha=0.7)
        
        # Прогноз
        plt.plot(self.forecast_df['time_index'], self.forecast_df['revenue_total'], 
                'r--', label='Прогноз', linewidth=2, marker='o')
        
        plt.title('Прогноз Revenue Total\n(Обучение на исторических данных)')
        plt.xlabel('Временной индекс')
        plt.ylabel('Revenue Total')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Подграфик 2: Распределение прогноза
        plt.subplot(2, 2, 2)
        plt.hist(self.forecast_df['revenue_total'], bins=50, alpha=0.7, color='red')
        plt.title('Распределение прогнозных значений')
        plt.xlabel('Revenue Total')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        
        # Подграфик 3: Топ-10 комбинаций
        plt.subplot(2, 2, 3)
        top_combinations = self.forecast_df.nlargest(10, 'revenue_total')
        if 'region_to' in top_combinations.columns and 'subdivision' in top_combinations.columns:
            labels = [f"{row['region_to']}\n{row['subdivision']}" for _, row in top_combinations.iterrows()]
            plt.bar(range(len(labels)), top_combinations['revenue_total'])
            plt.title('Топ-10 комбинаций по выручке')
            plt.xlabel('Комбинации')
            plt.ylabel('Revenue Total')
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # Подграфик 4: Сезонность
        plt.subplot(2, 2, 4)
        if 'month_sin' in self.forecast_df.columns and 'month_cos' in self.forecast_df.columns:
            # Восстанавливаем месяц из тригонометрических функций
            months = np.arctan2(self.forecast_df['month_sin'], self.forecast_df['month_cos']) * 6 / np.pi + 6
            months = np.where(months < 0, months + 12, months)
            monthly_revenue = self.forecast_df.groupby(months)['revenue_total'].sum()
            plt.bar(monthly_revenue.index, monthly_revenue.values)
            plt.title('Сезонность прогноза')
            plt.xlabel('Месяц')
            plt.ylabel('Revenue Total')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('historical_trend_forecast_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 График сохранен как 'historical_trend_forecast_analysis.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("🎯 ПРОГРАММА ПРОГНОЗИРОВАНИЯ С ОБУЧЕНИЕМ НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("="*70)
    
    # Инициализация
    forecaster = HistoricalTrendForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и анализ данных
    forecaster.load_and_analyze_data()
    
    # Очистка данных
    forecaster.clean_data()
    
    # Разделение на обучающую и тестовую выборки
    forecaster.split_historical_data(train_end_year=2025, train_end_month=8)
    
    # Обучение моделей
    forecaster.train_models()
    
    # Создание прогноза
    forecaster.create_forecast(forecast_periods=4)
    
    # Сохранение результатов
    forecaster.save_forecast()
    
    # Визуализация
    forecaster.plot_forecast_analysis()
    
    print(f"\n🎉 Программа завершена успешно!")

if __name__ == "__main__":
    main()
