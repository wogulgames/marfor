#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа для прогнозирования маркетинговых данных
Продлевает тренд до конца 2026 года на основе исторических данных
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MarketingForecast:
    def __init__(self, csv_file):
        """
        Инициализация класса прогнозирования
        
        Args:
            csv_file (str): Путь к CSV файлу с историческими данными
        """
        self.csv_file = csv_file
        self.df = None
        self.forecast_df = None
        
    def load_data(self):
        """Загрузка и предобработка данных"""
        print("Загрузка данных...")
        self.df = pd.read_csv(self.csv_file)
        
        # Очистка данных
        self.df = self.df.dropna(subset=['year', 'month'])
        self.df['year'] = self.df['year'].astype(int)
        self.df['month'] = self.df['month'].astype(int)
        
        # Очистка числовых колонок от запятых и преобразование в float
        numeric_columns = [
            'revenue_first_transactions', 'revenue_repeat_transactions', 'revenue_total',
            'first_traffic', 'repeat_traffic', 'traffic_total',
            'first_transactions', 'repeat_transactions', 'transacitons_total',
            'ads_cost', 'not_ads_cost', 'bonus_company', 'promo_cost', 'mar_cost',
            'Сумма по полю distributed_ads_cost'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                # Заменяем запятые на точки и преобразуем в float
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Создание временного индекса
        self.df['date'] = pd.to_datetime(self.df[['year', 'month']].assign(day=1))
        self.df['time_index'] = (self.df['year'] - 2023) * 12 + (self.df['month'] - 9)
        
        print(f"Загружено {len(self.df)} записей")
        print(f"Период данных: {self.df['date'].min()} - {self.df['date'].max()}")
        
    def prepare_forecast_data(self):
        """Подготовка данных для прогнозирования"""
        print("Подготовка данных для прогнозирования...")
        
        # Определяем последний месяц в данных
        last_date = self.df['date'].max()
        last_month = last_date.month
        last_year = last_date.year
        
        # Создаем список месяцев для прогнозирования (сентябрь-декабрь 2026)
        forecast_months = []
        for month in range(last_month + 1, 13):  # Следующие месяцы до конца года
            forecast_months.append({
                'year': last_year,
                'month': month,
                'date': pd.to_datetime(f'{last_year}-{month:02d}-01'),
                'time_index': (last_year - 2023) * 12 + (month - 9)
            })
        
        self.forecast_months = pd.DataFrame(forecast_months)
        print(f"Будет прогнозироваться {len(forecast_months)} месяцев")
        
    def forecast_metric(self, metric_column, group_columns=None, method='linear'):
        """
        Прогнозирование конкретной метрики
        
        Args:
            metric_column (str): Название колонки для прогнозирования
            group_columns (list): Колонки для группировки (например, ['subdivision', 'category'])
            method (str): Метод прогнозирования ('linear', 'polynomial')
        """
        print(f"Прогнозирование метрики: {metric_column}")
        
        if group_columns is None:
            group_columns = ['subdivision', 'category', 'is_paid']
        
        forecasts = []
        
        # Группируем данные
        grouped = self.df.groupby(group_columns)
        
        for group_key, group_data in grouped:
            if len(group_data) < 2:  # Минимум 2 точки для прогноза
                continue
                
            # Подготавливаем данные для модели
            X = group_data[['time_index']].values
            y = group_data[metric_column].fillna(0).values
            
            # Если все значения нулевые, пропускаем группу (не прогнозируем)
            if np.sum(y) == 0:
                print(f"Группа {group_key}: все значения нулевые, пропускаем прогноз")
                continue
            
            try:
                if method == 'linear':
                    model = LinearRegression()
                elif method == 'polynomial':
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    model = LinearRegression()
                    X = X_poly
                
                model.fit(X, y)
                
                # Прогнозируем для будущих месяцев
                X_future = self.forecast_months[['time_index']].values
                if method == 'polynomial':
                    X_future = poly_features.transform(X_future)
                
                y_pred = model.predict(X_future)
                
                # Создаем записи для прогноза
                for i, (_, forecast_row) in enumerate(self.forecast_months.iterrows()):
                    forecast_record = {
                        'year': forecast_row['year'],
                        'month': forecast_row['month'],
                        'Quarter': self._get_quarter(forecast_row['month']),
                        'Halfyear': self._get_halfyear(forecast_row['month']),
                        metric_column: max(0, y_pred[i])  # Не допускаем отрицательные значения
                    }
                    
                    # Добавляем группировочные колонки
                    if isinstance(group_key, tuple):
                        for j, col in enumerate(group_columns):
                            forecast_record[col] = group_key[j]
                    else:
                        forecast_record[group_columns[0]] = group_key
                    
                    forecasts.append(forecast_record)
                    
            except Exception as e:
                print(f"Ошибка при прогнозировании группы {group_key}: {e}")
                continue
        
        return pd.DataFrame(forecasts)
    
    def _get_quarter(self, month):
        """Определение квартала по месяцу"""
        if month in [1, 2, 3]:
            return 'Q1'
        elif month in [4, 5, 6]:
            return 'Q2'
        elif month in [7, 8, 9]:
            return 'Q3'
        else:
            return 'Q4'
    
    def _get_halfyear(self, month):
        """Определение полугодия по месяцу"""
        return 'H1' if month <= 6 else 'H2'
    
    def create_complete_forecast(self):
        """Создание полного прогноза для всех метрик"""
        print("Создание полного прогноза...")
        
        # Основные метрики для прогнозирования
        metrics_to_forecast = [
            'revenue_total',
            'revenue_first_transactions',
            'revenue_repeat_transactions',
            'traffic_total',
            'first_traffic',
            'repeat_traffic',
            'transacitons_total',
            'first_transactions',
            'repeat_transactions',
            'ads_cost',
            'not_ads_cost',
            'bonus_company',
            'promo_cost',
            'mar_cost',
            'Сумма по полю distributed_ads_cost'
        ]
        
        all_forecasts = []
        
        for metric in metrics_to_forecast:
            if metric in self.df.columns:
                forecast = self.forecast_metric(metric)
                if not forecast.empty:
                    all_forecasts.append(forecast)
        
        if all_forecasts:
            # Объединяем все прогнозы
            self.forecast_df = all_forecasts[0]
            for forecast in all_forecasts[1:]:
                # Удаляем дублирующиеся колонки перед объединением
                common_cols = ['year', 'month', 'subdivision', 'category', 'is_paid', 'Quarter', 'Halfyear']
                forecast_metrics = [col for col in forecast.columns if col not in common_cols]
                
                # Объединяем только метрики
                for metric in forecast_metrics:
                    if metric in forecast.columns:
                        self.forecast_df = self.forecast_df.merge(
                            forecast[common_cols + [metric]], 
                            on=common_cols, 
                            how='outer'
                        )
            
            # Заполняем недостающие колонки
            self._fill_missing_columns()
            
            # Исправляем логические несоответствия
            self._fix_logical_inconsistencies()
            
            print(f"Создан прогноз для {len(self.forecast_df)} записей")
        else:
            print("Не удалось создать прогноз")
    
    def _fill_missing_columns(self):
        """Заполнение недостающих колонок в прогнозе"""
        # Базовые колонки из исходных данных
        base_columns = [
            'region_to', 'is_global', 'KEY_clean', 'is_flowers',
            'revenue_first_transactions', 'revenue_repeat_transactions',
            'first_traffic', 'repeat_traffic', 'first_transactions', 
            'repeat_transactions', 'not_ads_cost', 'bonus_company', 'promo_cost'
        ]
        
        for col in base_columns:
            if col not in self.forecast_df.columns:
                if col in ['region_to', 'is_global', 'KEY_clean', 'is_flowers']:
                    # Для категориальных колонок берем наиболее частые значения
                    if col in self.df.columns:
                        most_common = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else ''
                        self.forecast_df[col] = most_common
                else:
                    # Для числовых колонок ставим 0
                    self.forecast_df[col] = 0
    
    def _fix_logical_inconsistencies(self):
        """Исправление логических несоответствий в прогнозе"""
        print("Исправление логических несоответствий...")
        
        # 1. Если есть трафик, но нет транзакций - рассчитываем конверсию
        traffic_without_transactions = self.forecast_df[
            (self.forecast_df['traffic_total'] > 0) & 
            (self.forecast_df['transacitons_total'] == 0)
        ]
        
        if len(traffic_without_transactions) > 0:
            print(f"Исправляем {len(traffic_without_transactions)} записей с трафиком без транзакций")
            
            # Рассчитываем среднюю конверсию из исторических данных
            historical_data = self.df[
                (self.df['traffic_total'] > 0) & 
                (self.df['transacitons_total'] > 0)
            ]
            
            if len(historical_data) > 0:
                avg_conversion = historical_data['transacitons_total'].sum() / historical_data['traffic_total'].sum()
                print(f"Средняя конверсия из исторических данных: {avg_conversion:.4f}")
                
                # Применяем конверсию к записям с трафиком без транзакций
                mask = (self.forecast_df['traffic_total'] > 0) & (self.forecast_df['transacitons_total'] == 0)
                self.forecast_df.loc[mask, 'transacitons_total'] = (
                    self.forecast_df.loc[mask, 'traffic_total'] * avg_conversion
                )
        
        # 2. Если есть транзакции, но нет first_transactions - распределяем пропорционально
        transactions_without_first = self.forecast_df[
            (self.forecast_df['transacitons_total'] > 0) & 
            (self.forecast_df['first_transactions'] == 0)
        ]
        
        if len(transactions_without_first) > 0:
            print(f"Исправляем {len(transactions_without_first)} записей с транзакциями без first_transactions")
            
            # Рассчитываем среднее соотношение first/repeat из исторических данных
            historical_with_both = self.df[
                (self.df['first_transactions'] > 0) & 
                (self.df['repeat_transactions'] > 0)
            ]
            
            if len(historical_with_both) > 0:
                avg_first_ratio = historical_with_both['first_transactions'].sum() / historical_with_both['transacitons_total'].sum()
                avg_repeat_ratio = historical_with_both['repeat_transactions'].sum() / historical_with_both['transacitons_total'].sum()
                
                print(f"Среднее соотношение first: {avg_first_ratio:.4f}, repeat: {avg_repeat_ratio:.4f}")
                
                # Применяем соотношения
                mask = (self.forecast_df['transacitons_total'] > 0) & (self.forecast_df['first_transactions'] == 0)
                self.forecast_df.loc[mask, 'first_transactions'] = (
                    self.forecast_df.loc[mask, 'transacitons_total'] * avg_first_ratio
                )
                self.forecast_df.loc[mask, 'repeat_transactions'] = (
                    self.forecast_df.loc[mask, 'transacitons_total'] * avg_repeat_ratio
                )
        
        # 3. Если есть выручка, но нет транзакций - рассчитываем средний чек
        revenue_without_transactions = self.forecast_df[
            (self.forecast_df['revenue_total'] > 0) & 
            (self.forecast_df['transacitons_total'] == 0)
        ]
        
        if len(revenue_without_transactions) > 0:
            print(f"Исправляем {len(revenue_without_transactions)} записей с выручкой без транзакций")
            
            # Рассчитываем средний чек из исторических данных
            historical_with_both = self.df[
                (self.df['revenue_total'] > 0) & 
                (self.df['transacitons_total'] > 0)
            ]
            
            if len(historical_with_both) > 0:
                avg_check = historical_with_both['revenue_total'].sum() / historical_with_both['transacitons_total'].sum()
                print(f"Средний чек из исторических данных: {avg_check:.2f}")
                
                # Применяем средний чек
                mask = (self.forecast_df['revenue_total'] > 0) & (self.forecast_df['transacitons_total'] == 0)
                self.forecast_df.loc[mask, 'transacitons_total'] = (
                    self.forecast_df.loc[mask, 'revenue_total'] / avg_check
                )
                
                # Также рассчитываем first и repeat транзакции
                mask = (self.forecast_df['revenue_total'] > 0) & (self.forecast_df['first_transactions'] == 0)
                if len(historical_with_both) > 0:
                    avg_first_ratio = historical_with_both['first_transactions'].sum() / historical_with_both['transacitons_total'].sum()
                    avg_repeat_ratio = historical_with_both['repeat_transactions'].sum() / historical_with_both['transacitons_total'].sum()
                    
                    self.forecast_df.loc[mask, 'first_transactions'] = (
                        self.forecast_df.loc[mask, 'transacitons_total'] * avg_first_ratio
                    )
                    self.forecast_df.loc[mask, 'repeat_transactions'] = (
                        self.forecast_df.loc[mask, 'transacitons_total'] * avg_repeat_ratio
                    )
        
        # 4. Исправляем недостающие поля на основе исторических соотношений
        self._fill_missing_fields_from_ratios()
        
        print("Логические несоответствия исправлены")
    
    def _fill_missing_fields_from_ratios(self):
        """Заполнение недостающих полей на основе исторических соотношений"""
        print("Заполнение недостающих полей на основе исторических соотношений...")
        
        # Получаем исторические данные с ненулевыми значениями
        historical_data = self.df[
            (self.df['revenue_total'] > 0) & 
            (self.df['traffic_total'] > 0) & 
            (self.df['transacitons_total'] > 0)
        ]
        
        if len(historical_data) == 0:
            print("Недостаточно исторических данных для расчета соотношений")
            return
        
        # Рассчитываем средние соотношения
        ratios = {}
        
        # Соотношения к общей выручке
        if historical_data['revenue_first_transactions'].sum() > 0:
            ratios['revenue_first_ratio'] = historical_data['revenue_first_transactions'].sum() / historical_data['revenue_total'].sum()
        if historical_data['revenue_repeat_transactions'].sum() > 0:
            ratios['revenue_repeat_ratio'] = historical_data['revenue_repeat_transactions'].sum() / historical_data['revenue_total'].sum()
        
        # Соотношения к общему трафику
        if historical_data['first_traffic'].sum() > 0:
            ratios['first_traffic_ratio'] = historical_data['first_traffic'].sum() / historical_data['traffic_total'].sum()
        if historical_data['repeat_traffic'].sum() > 0:
            ratios['repeat_traffic_ratio'] = historical_data['repeat_traffic'].sum() / historical_data['traffic_total'].sum()
        
        # Соотношения к общим транзакциям
        if historical_data['first_transactions'].sum() > 0:
            ratios['first_transactions_ratio'] = historical_data['first_transactions'].sum() / historical_data['transacitons_total'].sum()
        if historical_data['repeat_transactions'].sum() > 0:
            ratios['repeat_transactions_ratio'] = historical_data['repeat_transactions'].sum() / historical_data['transacitons_total'].sum()
        
        # Соотношения к маркетинговым расходам
        if historical_data['ads_cost'].sum() > 0:
            ratios['ads_cost_ratio'] = historical_data['ads_cost'].sum() / historical_data['mar_cost'].sum()
        if historical_data['not_ads_cost'].sum() > 0:
            ratios['not_ads_cost_ratio'] = historical_data['not_ads_cost'].sum() / historical_data['mar_cost'].sum()
        if historical_data['bonus_company'].sum() > 0:
            ratios['bonus_company_ratio'] = historical_data['bonus_company'].sum() / historical_data['mar_cost'].sum()
        if historical_data['promo_cost'].sum() > 0:
            ratios['promo_cost_ratio'] = historical_data['promo_cost'].sum() / historical_data['mar_cost'].sum()
        
        # Дополнительно проверим соотношения для всех исторических данных (не только с ненулевыми значениями)
        all_historical = self.df[
            (self.df['year'].isin([2023, 2024, 2025])) & 
            (self.df['mar_cost'] > 0)
        ]
        
        if len(all_historical) > 0:
            if all_historical['not_ads_cost'].sum() > 0:
                ratios['not_ads_cost_ratio_all'] = all_historical['not_ads_cost'].sum() / all_historical['mar_cost'].sum()
                print(f"Соотношение not_ads_cost / mar_cost (все исторические): {ratios['not_ads_cost_ratio_all']:.4f}")
        
        print(f"Рассчитанные соотношения: {ratios}")
        
        # Применяем соотношения к прогнозу
        forecast_data = self.forecast_df
        
        # Заполняем revenue_first_transactions и revenue_repeat_transactions
        if 'revenue_first_ratio' in ratios:
            mask = (forecast_data['revenue_total'] > 0) & (forecast_data['revenue_first_transactions'] == 0)
            forecast_data.loc[mask, 'revenue_first_transactions'] = (
                forecast_data.loc[mask, 'revenue_total'] * ratios['revenue_first_ratio']
            )
        
        if 'revenue_repeat_ratio' in ratios:
            mask = (forecast_data['revenue_total'] > 0) & (forecast_data['revenue_repeat_transactions'] == 0)
            forecast_data.loc[mask, 'revenue_repeat_transactions'] = (
                forecast_data.loc[mask, 'revenue_total'] * ratios['revenue_repeat_ratio']
            )
        
        # Заполняем first_traffic и repeat_traffic
        if 'first_traffic_ratio' in ratios:
            mask = (forecast_data['traffic_total'] > 0) & (forecast_data['first_traffic'] == 0)
            forecast_data.loc[mask, 'first_traffic'] = (
                forecast_data.loc[mask, 'traffic_total'] * ratios['first_traffic_ratio']
            )
        
        if 'repeat_traffic_ratio' in ratios:
            mask = (forecast_data['traffic_total'] > 0) & (forecast_data['repeat_traffic'] == 0)
            forecast_data.loc[mask, 'repeat_traffic'] = (
                forecast_data.loc[mask, 'traffic_total'] * ratios['repeat_traffic_ratio']
            )
        
        # Заполняем маркетинговые расходы
        if 'ads_cost_ratio' in ratios:
            mask = (forecast_data['mar_cost'] > 0) & (forecast_data['ads_cost'] == 0)
            forecast_data.loc[mask, 'ads_cost'] = (
                forecast_data.loc[mask, 'mar_cost'] * ratios['ads_cost_ratio']
            )
        
        if 'not_ads_cost_ratio' in ratios:
            mask = (forecast_data['mar_cost'] > 0) & (forecast_data['not_ads_cost'] == 0)
            forecast_data.loc[mask, 'not_ads_cost'] = (
                forecast_data.loc[mask, 'mar_cost'] * ratios['not_ads_cost_ratio']
            )
        
        # Применяем соотношение not_ads_cost из всех исторических данных
        if 'not_ads_cost_ratio_all' in ratios:
            mask = (forecast_data['mar_cost'] > 0) & (forecast_data['not_ads_cost'] == 0)
            forecast_data.loc[mask, 'not_ads_cost'] = (
                forecast_data.loc[mask, 'mar_cost'] * ratios['not_ads_cost_ratio_all']
            )
            print(f"Заполнено {mask.sum()} записей not_ads_cost на основе исторического соотношения {ratios['not_ads_cost_ratio_all']:.4f}")
        
        if 'bonus_company_ratio' in ratios:
            mask = (forecast_data['mar_cost'] > 0) & (forecast_data['bonus_company'] == 0)
            forecast_data.loc[mask, 'bonus_company'] = (
                forecast_data.loc[mask, 'mar_cost'] * ratios['bonus_company_ratio']
            )
        
        if 'promo_cost_ratio' in ratios:
            mask = (forecast_data['mar_cost'] > 0) & (forecast_data['promo_cost'] == 0)
            forecast_data.loc[mask, 'promo_cost'] = (
                forecast_data.loc[mask, 'mar_cost'] * ratios['promo_cost_ratio']
            )
        
        # Заполняем distributed_ads_cost на основе ads_cost
        if historical_data['Сумма по полю distributed_ads_cost'].sum() > 0 and historical_data['ads_cost'].sum() > 0:
            distributed_ratio = historical_data['Сумма по полю distributed_ads_cost'].sum() / historical_data['ads_cost'].sum()
            mask = (forecast_data['ads_cost'] > 0) & (forecast_data['Сумма по полю distributed_ads_cost'] == 0)
            forecast_data.loc[mask, 'Сумма по полю distributed_ads_cost'] = (
                forecast_data.loc[mask, 'ads_cost'] * distributed_ratio
            )
        
        print("Недостающие поля заполнены на основе исторических соотношений")
    
    def save_forecast(self, output_file):
        """Сохранение прогноза в CSV файл"""
        if self.forecast_df is not None:
            # Объединяем исходные данные с прогнозом
            combined_df = pd.concat([self.df, self.forecast_df], ignore_index=True)
            combined_df = combined_df.sort_values(['year', 'month', 'subdivision', 'category'])
            
            # Сохраняем в файл
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Прогноз сохранен в файл: {output_file}")
            print(f"Общее количество записей: {len(combined_df)}")
            print(f"Новых прогнозных записей: {len(self.forecast_df)}")
        else:
            print("Нет данных для сохранения")
    
    def generate_summary_report(self):
        """Генерация сводного отчета по прогнозу"""
        if self.forecast_df is None:
            print("Нет данных прогноза для отчета")
            return
        
        print("\n" + "="*50)
        print("СВОДНЫЙ ОТЧЕТ ПО ПРОГНОЗУ")
        print("="*50)
        
        # Общая статистика
        print(f"Период прогноза: {self.forecast_df['year'].min()}-{self.forecast_df['month'].min():02d} - {self.forecast_df['year'].max()}-{self.forecast_df['month'].max():02d}")
        print(f"Количество прогнозных записей: {len(self.forecast_df)}")
        
        # Топ категории по выручке
        if 'revenue_total' in self.forecast_df.columns:
            top_categories = self.forecast_df.groupby('category')['revenue_total'].sum().sort_values(ascending=False).head(10)
            print("\nТоп-10 категорий по прогнозируемой выручке:")
            for category, revenue in top_categories.items():
                print(f"  {category}: {revenue:,.0f}")
        
        # Топ подразделения по трафику
        if 'traffic_total' in self.forecast_df.columns:
            top_subdivisions = self.forecast_df.groupby('subdivision')['traffic_total'].sum().sort_values(ascending=False)
            print("\nПодразделения по прогнозируемому трафику:")
            for subdivision, traffic in top_subdivisions.items():
                print(f"  {subdivision}: {traffic:,.0f}")
        
        print("="*50)

def main():
    """Основная функция программы"""
    print("Программа прогнозирования маркетинговых данных")
    print("="*50)
    
    # Инициализация
    forecast = MarketingForecast('Marketing Budjet Emulation - raw2.csv')
    
    # Загрузка и подготовка данных
    forecast.load_data()
    forecast.prepare_forecast_data()
    
    # Создание прогноза
    forecast.create_complete_forecast()
    
    # Сохранение результатов
    forecast.save_forecast('Marketing_Budget_Forecast_Extended.csv')
    
    # Генерация отчета
    forecast.generate_summary_report()
    
    print("\nПрограмма завершена успешно!")

if __name__ == "__main__":
    main()
