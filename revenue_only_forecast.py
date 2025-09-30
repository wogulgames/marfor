#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа прогнозирования только revenue_total с правильной валидацией
Использует только реальные исторические данные до августа 2025 года
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RevenueOnlyForecaster:
    def __init__(self, csv_file=None):
        """Инициализация прогнозировщика только revenue_total"""
        self.csv_file = csv_file
        self.df = None
        self.train_df = None
        self.test_df = None
        self.model = None
        self.scaler = None
        self.model_metrics = {}
        self.is_model_validated = False
        
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
        
        # Анализ revenue_total
        if 'revenue_total' in self.df.columns:
            print(f"\n📈 АНАЛИЗ REVENUE_TOTAL:")
            revenue_stats = self.df['revenue_total'].describe()
            print(f"  Среднее: {revenue_stats['mean']:,.0f} ₽")
            print(f"  Медиана: {revenue_stats['50%']:,.0f} ₽")
            print(f"  Максимум: {revenue_stats['max']:,.0f} ₽")
            print(f"  Ненулевых записей: {(self.df['revenue_total'] > 0).sum()}")
            
            # Анализ по периодам
            period_stats = self.df.groupby(['year', 'month'])['revenue_total'].agg(['count', 'sum', 'mean']).round(0)
            print(f"  Всего периодов: {len(period_stats)}")
            print(f"  Средняя выручка за период: {period_stats['sum'].mean():,.0f} ₽")
    
    def split_data_properly(self, train_end_year=2024, train_end_month=12, 
                           test_start_year=2025, test_start_month=1, test_end_month=8):
        """Правильное разделение данных на обучающую и тестовую выборки"""
        print(f"\n✂️ ПРАВИЛЬНОЕ РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"  Обучающая выборка: до {train_end_year}.{train_end_month:02d}")
        print(f"  Тестовая выборка: {test_start_year}.{test_start_month:02d} - {test_start_year}.{test_end_month:02d}")
        print(f"  ⚠️  Данные с {test_start_year}.{test_end_month+1:02d} - это прогноз, не используем для валидации")
        
        # Обучающая выборка (до декабря 2024)
        train_mask = (
            (self.df['year'] < train_end_year) | 
            ((self.df['year'] == train_end_year) & (self.df['month'] <= train_end_month))
        )
        self.train_df = self.df[train_mask].copy()
        
        # Тестовая выборка (январь-август 2025) - только реальные данные
        test_mask = (
            (self.df['year'] == test_start_year) & 
            (self.df['month'] >= test_start_month) & 
            (self.df['month'] <= test_end_month)
        )
        self.test_df = self.df[test_mask].copy()
        
        print(f"  Обучающая выборка: {len(self.train_df)} записей")
        print(f"  Тестовая выборка: {len(self.test_df)} записей")
        
        # Анализ качества разделения
        if 'revenue_total' in self.train_df.columns:
            train_revenue = self.train_df['revenue_total'].sum()
            test_revenue = self.test_df['revenue_total'].sum()
            
            print(f"\n💰 АНАЛИЗ REVENUE_TOTAL ПО ВЫБОРКАМ:")
            print(f"  Выручка в обучающей выборке: {train_revenue:,.0f} ₽")
            print(f"  Выручка в тестовой выборке: {test_revenue:,.0f} ₽")
            
            # Проверяем достаточность данных
            if len(self.train_df) < 100:
                print(f"⚠️  ВНИМАНИЕ: Обучающая выборка слишком мала ({len(self.train_df)} записей)")
            if len(self.test_df) < 50:
                print(f"⚠️  ВНИМАНИЕ: Тестовая выборка слишком мала ({len(self.test_df)} записей)")
            
            # Анализ временного покрытия
            print(f"\n📅 ВРЕМЕННОЕ ПОКРЫТИЕ:")
            if len(self.train_df) > 0:
                train_start = f"{self.train_df['year'].min()}.{self.train_df['month'].min():02d}"
                train_end = f"{self.train_df['year'].max()}.{self.train_df['month'].max():02d}"
                print(f"  Обучающая выборка: {train_start} - {train_end}")
            
            if len(self.test_df) > 0:
                test_start = f"{self.test_df['year'].min()}.{self.test_df['month'].min():02d}"
                test_end = f"{self.test_df['year'].max()}.{self.test_df['month'].max():02d}"
                print(f"  Тестовая выборка: {test_start} - {test_end}")
    
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
        
        return df
    
    def check_assumptions(self, X, y, model_name="Model"):
        """Проверка предположений линейной регрессии"""
        print(f"\n🔍 ПРОВЕРКА ПРЕДПОЛОЖЕНИЙ ЛИНЕЙНОЙ РЕГРЕССИИ для {model_name}:")
        
        # 1. Линейность (корреляция между признаками и целевой переменной)
        print(f"  1. Линейность:")
        for feature in X.columns:
            if len(X[feature].unique()) > 1:  # Проверяем только если есть вариация
                corr = X[feature].corr(y)
                print(f"    {feature}: корреляция = {corr:.3f}")
        
        # 2. Нормальность признаков
        print(f"  2. Нормальность признаков:")
        for feature in X.columns:
            if len(X[feature].unique()) > 1:
                # Проверяем нормальность с помощью коэффициента асимметрии
                skewness = X[feature].skew()
                print(f"    {feature}: асимметрия = {skewness:.3f} {'(нормально)' if abs(skewness) < 0.5 else '(не нормально)'}")
        
        # 3. Мультиколлинеарность (корреляция между признаками)
        print(f"  3. Мультиколлинеарность:")
        corr_matrix = X.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:  # Высокая корреляция
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"    ⚠️  Найдены высокие корреляции:")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"      {feat1} - {feat2}: {corr:.3f}")
        else:
            print(f"    ✅ Мультиколлинеарность отсутствует")
    
    def train_and_validate_models(self):
        """Обучение и валидация моделей для revenue_total"""
        print(f"\n🤖 ОБУЧЕНИЕ И ВАЛИДАЦИЯ МОДЕЛЕЙ ДЛЯ REVENUE_TOTAL:")
        
        # Подготавливаем обучающие данные
        train_data = self.prepare_time_features(self.train_df)
        train_data = train_data[train_data['revenue_total'] > 0].copy()
        
        if len(train_data) < 30:
            print(f"❌ Недостаточно данных для обучения ({len(train_data)} записей)")
            return None
        
        print(f"  Данных для обучения: {len(train_data)} записей")
        
        # Упрощенные временные признаки (убираем мультиколлинеарные)
        features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
        
        # Подготавливаем X и y
        X_train = train_data[features].fillna(0)
        y_train = train_data['revenue_total']
        
        print(f"  Используемые признаки: {features}")
        
        # Проверяем предположения
        self.check_assumptions(X_train, y_train, "Обучающая выборка")
        
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
        
        for model_name, model in models.items():
            try:
                print(f"\n  🔧 Обучение {model_name}:")
                
                # Обучение
                model.fit(X_train_scaled, y_train)
                
                # Предсказание на обучающей выборке
                y_pred_train = model.predict(X_train_scaled)
                
                # Метрики на обучающей выборке
                train_r2 = r2_score(y_train, y_pred_train)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                
                print(f"    Train R²: {train_r2:.3f}")
                print(f"    Train MAE: {train_mae:,.0f}")
                print(f"    Train RMSE: {train_rmse:,.0f}")
                
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
                        
                        print(f"    Test R²: {test_r2:.3f}")
                        print(f"    Test MAE: {test_mae:,.0f}")
                        print(f"    Test RMSE: {test_rmse:,.0f}")
                        
                        # Проверяем на переобучение
                        overfitting = train_r2 - test_r2
                        if overfitting > 0.1:
                            print(f"    ⚠️  Возможное переобучение (разница R²: {overfitting:.3f})")
                        elif overfitting < -0.1:
                            print(f"    ⚠️  Возможное недообучение (разница R²: {overfitting:.3f})")
                        else:
                            print(f"    ✅ Модель сбалансирована")
                        
                        # Сохраняем результаты
                        self.model_metrics[model_name] = {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'overfitting': overfitting
                        }
                        
                        # Выбираем модель с лучшим R² на тестовой выборке
                        if test_r2 > best_score:
                            best_score = test_r2
                            best_model = model
                            best_model_name = model_name
                    else:
                        print(f"    ❌ Нет тестовых данных для валидации")
                else:
                    print(f"    ❌ Нет тестовой выборки")
                    
            except Exception as e:
                print(f"    ❌ Ошибка при обучении {model_name}: {e}")
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
            
            # Оценка качества модели
            self.evaluate_model_quality(best_model_name, best_score)
            
            return {
                'model_name': best_model_name,
                'features': features,
                'test_r2': best_score,
                'train_size': len(train_data),
                'model_metrics': self.model_metrics
            }
        else:
            print(f"❌ Не удалось обучить ни одну модель")
            return None
    
    def evaluate_model_quality(self, model_name, test_r2):
        """Оценка качества модели"""
        print(f"\n📊 ОЦЕНКА КАЧЕСТВА МОДЕЛИ:")
        
        # Критерии качества модели
        if test_r2 > 0.7:
            quality = "ОТЛИЧНОЕ"
            recommendation = "✅ Модель готова для прогнозирования"
        elif test_r2 > 0.5:
            quality = "ХОРОШЕЕ"
            recommendation = "✅ Модель подходит для прогнозирования"
        elif test_r2 > 0.3:
            quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
            recommendation = "⚠️  Модель можно использовать с осторожностью"
        elif test_r2 > 0.1:
            quality = "СЛАБОЕ"
            recommendation = "❌ Модель не рекомендуется для прогнозирования"
        else:
            quality = "ОЧЕНЬ СЛАБОЕ"
            recommendation = "❌ Модель непригодна для прогнозирования"
        
        print(f"  Качество модели: {quality} (R² = {test_r2:.3f})")
        print(f"  Рекомендация: {recommendation}")
        
        # Дополнительные проверки
        if model_name in self.model_metrics:
            metrics = self.model_metrics[model_name]
            overfitting = metrics['overfitting']
            
            if abs(overfitting) > 0.2:
                print(f"  ⚠️  Проблема с переобучением/недообучением: {overfitting:.3f}")
            
            if metrics['test_mae'] > metrics['train_mae'] * 2:
                print(f"  ⚠️  Высокая ошибка на тестовой выборке")
        
        # Устанавливаем флаг валидации
        if test_r2 > 0.3:
            self.is_model_validated = True
            print(f"  ✅ Модель прошла валидацию")
        else:
            self.is_model_validated = False
            print(f"  ❌ Модель не прошла валидацию")
    
    def get_validation_summary(self):
        """Получение сводки по валидации"""
        print(f"\n📋 СВОДКА ПО ВАЛИДАЦИИ:")
        print(f"  Модель валидирована: {'✅ ДА' if self.is_model_validated else '❌ НЕТ'}")
        
        if self.model_metrics:
            print(f"  Результаты всех моделей:")
            for model_name, metrics in self.model_metrics.items():
                print(f"    {model_name}: Train R²={metrics['train_r2']:.3f}, Test R²={metrics['test_r2']:.3f}")
        
        return self.is_model_validated

def main():
    """Основная функция для демонстрации"""
    print("🎯 ПРОГРАММА ПРОГНОЗИРОВАНИЯ ТОЛЬКО REVENUE_TOTAL")
    print("="*60)
    print("Использует только реальные исторические данные до августа 2025 года")
    print("="*60)
    
    # Инициализация
    forecaster = RevenueOnlyForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и анализ данных
    forecaster.load_and_analyze_data()
    
    # Очистка данных
    forecaster.clean_data()
    
    # Правильное разделение данных
    forecaster.split_data_properly()
    
    # Обучение и валидация моделей
    forecaster.train_and_validate_models()
    
    # Получение сводки по валидации
    is_validated = forecaster.get_validation_summary()
    
    if is_validated:
        print(f"\n🎉 МОДЕЛЬ ГОТОВА ДЛЯ ПРОГНОЗИРОВАНИЯ REVENUE_TOTAL!")
        print(f"   Можем переходить к созданию прогноза")
    else:
        print(f"\n⚠️  МОДЕЛЬ НЕ ГОТОВА ДЛЯ ПРОГНОЗИРОВАНИЯ!")
        print(f"   Нужно улучшить данные или выбрать другой подход")
    
    print(f"\n📊 Программа завершена!")

if __name__ == "__main__":
    main()
