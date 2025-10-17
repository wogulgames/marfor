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
        
        # Специальные признаки на основе lag_12 (годовая сезонность)
        if 12 in lags:
            # Отношение текущего значения к тому же месяцу прошлого года
            self.df[f'{self.metric}_yoy_ratio'] = self.df[self.metric] / (self.df[f'{self.metric}_lag_12'] + 1)
            
            # Разница с прошлым годом
            self.df[f'{self.metric}_yoy_diff'] = self.df[self.metric] - self.df[f'{self.metric}_lag_12']
        
        # Количество добавленных признаков
        added_count = len(lags) + (2 if 12 in lags else 0)
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
    
    def add_seasonal_features(self, auto_detect_peaks=True, peak_threshold=1.2):
        """
        Добавляет сезонные признаки
        
        Создает:
        - sin/cos для месяца (годовая сезонность) - для плавной сезонности
        - month dummy variables - для учета специфики каждого месяца
        - is_peak_month - автоматически определяется из исторических данных
        
        Args:
            auto_detect_peaks: автоматически определять пиковые месяцы из данных
            peak_threshold: порог для определения пика (1.2 = на 20% выше среднего)
        """
        print(f"   📊 Добавление сезонных признаков", flush=True)
        
        # 1. Синусоиды для плавной сезонности
        self.df['month_sin'] = np.sin(2 * np.pi * self.df[self.time_col] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df[self.time_col] / 12)
        
        # 2. One-hot encoding для месяцев (более точный учет сезонности)
        for month in range(1, 13):
            self.df[f'is_month_{month}'] = (self.df[self.time_col] == month).astype(int)
        
        # 3. Автоматическое определение пиковых месяцев
        if auto_detect_peaks:
            # Вычисляем среднее значение по каждому месяцу
            month_avg = self.df.groupby(self.time_col)[self.metric].mean()
            overall_avg = self.df[self.metric].mean()
            
            # Месяцы, где среднее значение > порога считаются пиковыми
            peak_months = [int(month) for month, avg in month_avg.items() if avg > overall_avg * peak_threshold]
            
            if peak_months:
                print(f"      📈 Автоматически определены пиковые месяцы: {peak_months}", flush=True)
                print(f"      📊 Средние по месяцам: {dict(month_avg)}", flush=True)
            else:
                # Fallback на стандартные праздники
                peak_months = [2, 3, 5, 11, 12]
                print(f"      ⚠️ Пиковые месяцы не определены, используем стандартные: {peak_months}", flush=True)
        else:
            peak_months = [2, 3, 5, 11, 12]
        
        self.df['is_peak_month'] = self.df[self.time_col].isin(peak_months).astype(int)
        
        # Сохраняем пиковые месяцы для использования при прогнозе
        self.peak_months = peak_months
        
        # 4. Квартал (Q4 обычно самый сильный)
        self.df['is_q4'] = ((self.df[self.time_col] >= 10) & (self.df[self.time_col] <= 12)).astype(int)
        
        print(f"   ✅ Добавлено сезонных признаков: 18 (2 синусоиды + 12 месяцев + 2 пиковых + 2 квартал)", flush=True)
        
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
            # 1. Синусоиды
            row['month_sin'] = np.sin(2 * np.pi * period['month'] / 12)
            row['month_cos'] = np.cos(2 * np.pi * period['month'] / 12)
            
            # 2. One-hot encoding для месяцев
            for month in range(1, 13):
                row[f'is_month_{month}'] = 1 if period['month'] == month else 0
            
            # 3. Пиковые месяцы
            peak_months = [2, 3, 5, 11, 12]
            row['is_peak_month'] = 1 if period['month'] in peak_months else 0
            
            # 4. Q4 признак
            row['is_q4'] = 1 if period['month'] >= 10 else 0
            
            # Трендовые признаки
            row['time_index_squared'] = row['time_index'] ** 2
            
            future_rows.append(row)
        
        return pd.DataFrame(future_rows)

