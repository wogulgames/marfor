#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальная программа прогнозирования временных рядов
с валидацией модели и множественной регрессией
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class UniversalForecaster:
    def __init__(self, csv_file=None, date_columns=None, target_columns=None):
        """
        Инициализация универсального прогнозировщика
        
        Args:
            csv_file (str): Путь к CSV файлу
            date_columns (list): Список колонок с датами/временем
            target_columns (list): Список колонок для прогнозирования
        """
        self.csv_file = csv_file
        self.date_columns = date_columns or ['year', 'month']
        self.target_columns = target_columns
        self.df = None
        self.forecast_df = None
        self.models = {}
        self.scalers = {}
        self.validation_results = {}
        self.feature_importance = {}
        
    def load_data(self, csv_file=None):
        """Загрузка данных из CSV файла"""
        if csv_file:
            self.csv_file = csv_file
            
        if not self.csv_file:
            raise ValueError("Не указан файл для загрузки")
            
        print(f"Загрузка данных из {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"Загружено {len(self.df)} записей, {len(self.df.columns)} колонок")
        
        return self.df
    
    def auto_detect_columns(self):
        """Автоматическое определение колонок для прогнозирования"""
        print("Автоматическое определение колонок...")
        
        # Определяем временные колонки
        time_keywords = ['year', 'month', 'date', 'time', 'period', 'quarter']
        detected_time_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                detected_time_cols.append(col)
        
        if detected_time_cols:
            self.date_columns = detected_time_cols
            print(f"Обнаружены временные колонки: {self.date_columns}")
        
        # Определяем числовые колонки для прогнозирования
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Исключаем временные колонки и ID колонки
        exclude_keywords = ['id', 'index', 'key', 'code'] + [col.lower() for col in self.date_columns]
        target_candidates = []
        
        for col in numeric_cols:
            col_lower = col.lower()
            if not any(keyword in col_lower for keyword in exclude_keywords):
                # Проверяем, что в колонке есть не только нули
                if self.df[col].sum() > 0:
                    target_candidates.append(col)
        
        self.target_columns = target_candidates
        print(f"Обнаружены колонки для прогнозирования: {self.target_columns}")
        
        return self.target_columns
    
    def prepare_time_features(self):
        """Подготовка временных признаков"""
        print("Подготовка временных признаков...")
        
        # Очистка временных колонок
        for col in self.date_columns:
            if col in self.df.columns:
                if col in ['year', 'month']:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                else:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Создаем временной индекс
        if 'year' in self.df.columns and 'month' in self.df.columns:
            self.df['time_index'] = (self.df['year'] - self.df['year'].min()) * 12 + (self.df['month'] - 1)
        else:
            # Если нет year/month, используем порядковый номер
            self.df = self.df.sort_values(self.date_columns)
            self.df['time_index'] = range(len(self.df))
        
        # Сезонные признаки
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            self.df['quarter'] = ((self.df['month'] - 1) // 3) + 1
            
            # Квартальные dummy переменные
            for q in range(1, 5):
                self.df[f'q{q}'] = (self.df['quarter'] == q).astype(int)
        
        # Праздничные периоды (примерные)
        if 'month' in self.df.columns:
            self.df['holiday_period'] = (
                (self.df['month'] == 12) |  # Декабрь
                (self.df['month'] == 1) |   # Январь
                (self.df['month'] == 2) |   # Февраль
                (self.df['month'] == 3) |   # Март
                (self.df['month'] == 5)     # Май
            ).astype(int)
        
        print("Временные признаки подготовлены")
    
    def create_interaction_features(self, df):
        """Создание признаков взаимодействия"""
        # Логарифмические признаки
        for col in self.target_columns:
            if col in df.columns and df[col].sum() > 0:
                df[f'{col}_log'] = np.log1p(df[col])
        
        # Взаимодействие времени с сезонностью
        if 'time_index' in df.columns and 'month_sin' in df.columns:
            df['time_seasonal'] = df['time_index'] * df['month_sin']
        
        # Полиномиальные признаки времени
        if 'time_index' in df.columns:
            df['time_squared'] = df['time_index'] ** 2
            df['time_cubed'] = df['time_index'] ** 3
        
        return df
    
    def prepare_features(self, df):
        """Подготовка всех признаков для модели"""
        df = df.copy()
        df = self.create_interaction_features(df)
        
        # Базовые признаки
        base_features = ['time_index', 'month_sin', 'month_cos']
        
        # Добавляем квартальные признаки
        for q in range(1, 5):
            if f'q{q}' in df.columns:
                base_features.append(f'q{q}')
        
        # Добавляем праздничные периоды
        if 'holiday_period' in df.columns:
            base_features.append('holiday_period')
        
        # Добавляем полиномиальные признаки
        if 'time_squared' in df.columns:
            base_features.extend(['time_squared', 'time_cubed'])
        
        if 'time_seasonal' in df.columns:
            base_features.append('time_seasonal')
        
        # Добавляем связанные метрики как признаки
        for target in self.target_columns:
            if target in df.columns:
                base_features.append(target)
                if f'{target}_log' in df.columns:
                    base_features.append(f'{target}_log')
        
        # Убираем дубликаты
        base_features = list(set(base_features))
        
        # Проверяем наличие колонок
        available_features = [col for col in base_features if col in df.columns]
        
        return available_features
    
    def train_models_with_validation(self, test_size=0.2, cv_folds=5):
        """Обучение моделей с валидацией"""
        print("Обучение моделей с валидацией...")
        
        # Подготавливаем данные
        self.prepare_time_features()
        
        # Создаем признаки взаимодействия в основном датафрейме
        self.df = self.create_interaction_features(self.df)
        
        available_features = self.prepare_features(self.df)
        
        print(f"Используемые признаки: {available_features}")
        
        for target in self.target_columns:
            if target not in self.df.columns:
                continue
                
            print(f"\nОбучение модели для {target}...")
            
            # Подготавливаем данные для обучения
            train_data = self.df[self.df[target] > 0].copy()
            
            if len(train_data) < 20:  # Минимум 20 записей
                print(f"  Недостаточно данных для {target}, пропускаем")
                continue
            
            # Подготавливаем X и y
            X = train_data[available_features].fillna(0)
            y = train_data[target]
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Нормализация
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Обучение нескольких моделей
            models_to_try = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = -np.inf
            best_model_name = None
            
            for model_name, model in models_to_try.items():
                try:
                    # Обучение
                    model.fit(X_train_scaled, y_train)
                    
                    # Предсказание
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Метрики
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Выбираем модель с лучшим R² на тестовой выборке
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_model_name = model_name
                    
                    print(f"  {model_name}:")
                    print(f"    Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
                    print(f"    Train MAE: {train_mae:.0f}, Test MAE: {test_mae:.0f}")
                    
                except Exception as e:
                    print(f"  Ошибка при обучении {model_name}: {e}")
                    continue
            
            if best_model is not None:
                # Сохраняем лучшую модель
                self.models[target] = {
                    'model': best_model,
                    'model_name': best_model_name,
                    'features': available_features,
                    'test_r2': best_score,
                    'train_size': len(train_data)
                }
                self.scalers[target] = scaler
                
                # Валидация с временными рядами
                self._time_series_validation(target, X, y, available_features, scaler)
                
                print(f"  Лучшая модель: {best_model_name} (R² = {best_score:.3f})")
            else:
                print(f"  Не удалось обучить модель для {target}")
    
    def _time_series_validation(self, target, X, y, features, scaler):
        """Валидация модели на временных рядах"""
        print(f"  Валидация временных рядов для {target}...")
        
        # Используем TimeSeriesSplit для валидации
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Нормализация
            X_train_scaled = scaler.fit_transform(X_train_cv)
            X_val_scaled = scaler.transform(X_val_cv)
            
            # Обучение модели
            model = self.models[target]['model']
            model.fit(X_train_scaled, y_train_cv)
            
            # Предсказание
            y_pred_cv = model.predict(X_val_scaled)
            
            # Метрика
            r2_cv = r2_score(y_val_cv, y_pred_cv)
            cv_scores.append(r2_cv)
        
        avg_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"    Cross-validation R²: {avg_cv_score:.3f} ± {std_cv_score:.3f}")
        
        # Сохраняем результаты валидации
        self.validation_results[target] = {
            'cv_scores': cv_scores,
            'avg_cv_score': avg_cv_score,
            'std_cv_score': std_cv_score
        }
    
    def create_forecast(self, forecast_periods=4):
        """Создание прогноза"""
        print(f"Создание прогноза на {forecast_periods} периодов...")
        
        if not self.models:
            raise ValueError("Модели не обучены. Сначала выполните train_models_with_validation()")
        
        # Определяем последний временной индекс
        last_time_index = self.df['time_index'].max()
        
        # Создаем периоды для прогноза
        forecast_periods_data = []
        for i in range(1, forecast_periods + 1):
            period_data = {
                'time_index': last_time_index + i,
                'month_sin': np.sin(2 * np.pi * ((last_time_index + i) % 12) / 12),
                'month_cos': np.cos(2 * np.pi * ((last_time_index + i) % 12) / 12),
            }
            
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
            period_data['time_seasonal'] = period_data['time_index'] * period_data['month_sin']
            
            # Добавляем средние значения связанных метрик из исторических данных
            for target in self.target_columns:
                if target in self.df.columns:
                    avg_value = self.df[self.df[target] > 0][target].mean()
                    period_data[target] = avg_value
                    period_data[f'{target}_log'] = np.log1p(avg_value)
            
            forecast_periods_data.append(period_data)
        
        self.forecast_df = pd.DataFrame(forecast_periods_data)
        
        # Прогнозируем каждую метрику
        for target, model_info in self.models.items():
            try:
                # Подготавливаем признаки для прогноза
                forecast_features = self.forecast_df[model_info['features']].fillna(0)
                
                # Нормализация
                forecast_scaled = self.scalers[target].transform(forecast_features)
                
                # Прогноз
                predictions = model_info['model'].predict(forecast_scaled)
                
                # Сохраняем прогноз
                self.forecast_df[target] = np.maximum(0, predictions)  # Не допускаем отрицательные значения
                
                print(f"  {target}: прогноз создан")
                
            except Exception as e:
                print(f"  Ошибка при прогнозировании {target}: {e}")
                self.forecast_df[target] = 0
        
        print(f"Прогноз создан для {len(forecast_periods_data)} периодов")
    
    def save_forecast(self, output_file='Universal_Forecast_Results.csv'):
        """Сохранение результатов прогноза"""
        if self.forecast_df is None:
            raise ValueError("Прогноз не создан. Сначала выполните create_forecast()")
        
        # Сохраняем прогноз
        self.forecast_df.to_csv(output_file, index=False)
        print(f"Прогноз сохранен в файл: {output_file}")
        
        return self.forecast_df
    
    def generate_model_report(self):
        """Генерация отчета о качестве моделей"""
        print("\n" + "="*60)
        print("ОТЧЕТ О КАЧЕСТВЕ МОДЕЛЕЙ")
        print("="*60)
        
        for target, model_info in self.models.items():
            print(f"\n{target}:")
            print(f"  Модель: {model_info['model_name']}")
            print(f"  Тестовый R²: {model_info['test_r2']:.3f}")
            print(f"  Размер обучающей выборки: {model_info['train_size']}")
            
            if target in self.validation_results:
                cv_info = self.validation_results[target]
                print(f"  Cross-validation R²: {cv_info['avg_cv_score']:.3f} ± {cv_info['std_cv_score']:.3f}")
        
        print("="*60)
    
    def plot_forecast(self, target_columns=None, save_plot=True):
        """Визуализация прогноза"""
        if target_columns is None:
            target_columns = list(self.models.keys())[:4]  # Показываем первые 4 метрики
        
        n_plots = len(target_columns)
        if n_plots == 0:
            print("Нет метрик для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, target in enumerate(target_columns):
            if i >= 4:  # Максимум 4 графика
                break
                
            if target not in self.models:
                continue
            
            ax = axes[i]
            
            # Исторические данные
            historical_data = self.df[self.df[target] > 0]
            if len(historical_data) > 0:
                ax.plot(historical_data['time_index'], historical_data[target], 
                       'b-', label='Исторические данные', linewidth=2)
            
            # Прогноз
            if target in self.forecast_df.columns:
                ax.plot(self.forecast_df['time_index'], self.forecast_df[target], 
                       'r--', label='Прогноз', linewidth=2, marker='o')
            
            ax.set_title(f'{target}')
            ax.set_xlabel('Временной индекс')
            ax.set_ylabel('Значение')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Скрываем лишние подграфики
        for i in range(n_plots, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('forecast_visualization.png', dpi=300, bbox_inches='tight')
            print("График сохранен как 'forecast_visualization.png'")
        
        plt.show()

def main():
    """Основная функция для демонстрации"""
    print("Универсальная программа прогнозирования")
    print("="*50)
    
    # Инициализация
    forecaster = UniversalForecaster('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка данных
    forecaster.load_data()
    
    # Автоматическое определение колонок
    forecaster.auto_detect_columns()
    
    # Обучение моделей с валидацией
    forecaster.train_models_with_validation()
    
    # Создание прогноза
    forecaster.create_forecast(forecast_periods=4)
    
    # Сохранение результатов
    forecaster.save_forecast()
    
    # Генерация отчета
    forecaster.generate_model_report()
    
    # Визуализация
    forecaster.plot_forecast()
    
    print("\nПрограмма завершена успешно!")

if __name__ == "__main__":
    main()