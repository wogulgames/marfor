"""
Модуль для создания расширенных признаков (features) для прогнозирования временных рядов.

Включает:
- Лаги (lag features): прошлые значения метрики
- Rolling features: скользящие статистики
- Сезонные признаки: синусоиды для учета сезонности
"""

import pandas as pd
import numpy as np


class FeatureBuilder:
    """Класс для построения признаков временных рядов"""
    
    def __init__(self, df, metric, time_col='month', year_col='year'):
        """
        Инициализация построителя признаков
        
        Args:
            df: DataFrame с данными
            metric: название целевой метрики
            time_col: название колонки с месяцем
            year_col: название колонки с годом
        """
        self.df = df.copy()
        self.metric = metric
        self.time_col = time_col
        self.year_col = year_col
        
        # Создаем временной индекс для правильного расчета лагов
        self.df['time_index'] = (self.df[year_col] - self.df[year_col].min()) * 12 + self.df[time_col]
        self.df = self.df.sort_values('time_index')
    
    def add_lag_features(self, lags=[1, 2, 3, 4, 6, 12]):
        """
        Добавляет лаговые признаки
        
        Args:
            lags: список лагов (по умолчанию [1, 2, 3, 4, 6, 12])
                  1 = прошлый месяц
                  12 = тот же месяц прошлого года
        """
        print(f"   📊 Добавление лагов: {lags}", flush=True)
        
        for lag in lags:
            self.df[f'{self.metric}_lag_{lag}'] = self.df[self.metric].shift(lag)
        
        # Количество добавленных признаков
        added_count = len(lags)
        print(f"   ✅ Добавлено лаговых признаков: {added_count}", flush=True)
        
        return self
    
    def add_rolling_features(self, windows=[3, 6]):
        """
        Добавляет rolling (скользящие) признаки
        
        Args:
            windows: список размеров окон (по умолчанию [3, 6])
        """
        print(f"   📊 Добавление rolling-признаков: окна {windows}", flush=True)
        
        for window in windows:
            # Скользящее среднее
            self.df[f'{self.metric}_rolling_mean_{window}'] = self.df[self.metric].rolling(window=window).mean()
            
            # Скользящее стандартное отклонение
            self.df[f'{self.metric}_rolling_std_{window}'] = self.df[self.metric].rolling(window=window).std()
            
            # Скользящий минимум
            self.df[f'{self.metric}_rolling_min_{window}'] = self.df[self.metric].rolling(window=window).min()
            
            # Скользящий максимум
            self.df[f'{self.metric}_rolling_max_{window}'] = self.df[self.metric].rolling(window=window).max()
        
        added_count = len(windows) * 4  # mean, std, min, max для каждого окна
        print(f"   ✅ Добавлено rolling-признаков: {added_count}", flush=True)
        
        return self
    
    def add_seasonal_features(self):
        """
        Добавляет сезонные признаки (синусоиды)
        
        Создает:
        - sin/cos для месяца (годовая сезонность)
        - sin/cos для квартала (квартальная сезонность)
        """
        print(f"   📊 Добавление сезонных признаков (синусоиды)", flush=True)
        
        # Годовая сезонность (месяц)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df[self.time_col] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df[self.time_col] / 12)
        
        # Квартальная сезонность
        self.df['quarter_sin'] = np.sin(2 * np.pi * ((self.df[self.time_col] - 1) // 3) / 4)
        self.df['quarter_cos'] = np.cos(2 * np.pi * ((self.df[self.time_col] - 1) // 3) / 4)
        
        print(f"   ✅ Добавлено сезонных признаков: 4", flush=True)
        
        return self
    
    def add_trend_features(self):
        """
        Добавляет признаки тренда
        
        Создает:
        - time_index: линейный тренд
        - time_index_squared: квадратичный тренд
        """
        print(f"   📊 Добавление трендовых признаков", flush=True)
        
        # Линейный тренд (уже есть в time_index)
        # Квадратичный тренд
        self.df['time_index_squared'] = self.df['time_index'] ** 2
        
        print(f"   ✅ Добавлено трендовых признаков: 2", flush=True)
        
        return self
    
    def add_interaction_features(self, categorical_cols=[]):
        """
        Добавляет признаки взаимодействия (interaction features)
        
        Args:
            categorical_cols: список категориальных колонок для взаимодействия с временем
        """
        if not categorical_cols:
            return self
        
        print(f"   📊 Добавление interaction-признаков: {categorical_cols}", flush=True)
        
        # Взаимодействие категорий с месяцем (для учета сезонности по категориям)
        for col in categorical_cols:
            if f'{col}_encoded' in self.df.columns:
                self.df[f'{col}_x_month'] = self.df[f'{col}_encoded'] * self.df[self.time_col]
        
        added_count = len([c for c in categorical_cols if f'{c}_encoded' in self.df.columns])
        print(f"   ✅ Добавлено interaction-признаков: {added_count}", flush=True)
        
        return self
    
    def get_feature_columns(self):
        """
        Возвращает список всех созданных признаков
        """
        # Все колонки, кроме целевой метрики и вспомогательных
        exclude_cols = [self.metric, 'time_index']
        
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols 
                       and not col.startswith('Unnamed')]
        
        return feature_cols
    
    def build_all_features(self, categorical_cols=[]):
        """
        Создает все признаки одним вызовом
        
        Args:
            categorical_cols: список категориальных колонок
        
        Returns:
            DataFrame с добавленными признаками
        """
        print(f"\n🔧 === ПОСТРОЕНИЕ ПРИЗНАКОВ ===", flush=True)
        
        self.add_lag_features([1, 2, 3, 4, 6, 12])
        self.add_rolling_features([3, 6])
        self.add_seasonal_features()
        self.add_trend_features()
        self.add_interaction_features(categorical_cols)
        
        feature_cols = self.get_feature_columns()
        print(f"\n✅ Всего создано признаков: {len(feature_cols)}", flush=True)
        print(f"   Список признаков: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"   Список признаков: {feature_cols}", flush=True)
        
        return self.df, feature_cols
    
    def prepare_for_prediction(self, future_periods, categorical_values=None):
        """
        Подготавливает признаки для будущих периодов (прогноз)
        
        Args:
            future_periods: список словарей с year, month
            categorical_values: словарь с значениями категориальных признаков
        
        Returns:
            DataFrame с признаками для будущих периодов
        """
        future_rows = []
        
        for period in future_periods:
            row = {
                self.year_col: period['year'],
                self.time_col: period['month']
            }
            
            # Добавляем категориальные признаки
            if categorical_values:
                row.update(categorical_values)
            
            # Временной индекс
            row['time_index'] = (period['year'] - self.df[self.year_col].min()) * 12 + period['month']
            
            # Сезонные признаки
            row['month_sin'] = np.sin(2 * np.pi * period['month'] / 12)
            row['month_cos'] = np.cos(2 * np.pi * period['month'] / 12)
            row['quarter_sin'] = np.sin(2 * np.pi * ((period['month'] - 1) // 3) / 4)
            row['quarter_cos'] = np.cos(2 * np.pi * ((period['month'] - 1) // 3) / 4)
            
            # Трендовые признаки
            row['time_index_squared'] = row['time_index'] ** 2
            
            future_rows.append(row)
        
        return pd.DataFrame(future_rows)

