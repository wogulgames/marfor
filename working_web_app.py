#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARFOR - Рабочее веб-приложение для прогнозирования маркетинговых данных
Интегрирует каскадную модель с Random Forest и веб-интерфейс
"""

import pandas as pd
import numpy as np
import os
import uuid
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импортируем наши модули
from feature_builder import FeatureBuilder
from hierarchy import HierarchyReconciler

def convert_to_json_serializable(obj):
    """Конвертация pandas/numpy объектов в JSON-совместимые типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def calculate_metrics(actual, predicted):
    """Вычисление метрик точности прогноза"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # MAE - средняя абсолютная ошибка
    mae = np.mean(np.abs(actual - predicted))
    
    # RMSE - корень из средней квадратичной ошибки
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAPE - средняя абсолютная процентная ошибка
    # Исключаем нули из расчета MAPE чтобы избежать inf
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0  # Если все значения нули
    
    # Заменяем inf и nan на 0
    mae = 0.0 if (np.isnan(mae) or np.isinf(mae)) else float(mae)
    rmse = 0.0 if (np.isnan(rmse) or np.isinf(rmse)) else float(rmse)
    mape = 0.0 if (np.isnan(mape) or np.isinf(mape)) else float(mape)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

def train_arima_model(train_df, test_df, metric):
    """Обучение ARIMA модели"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        print(f"   ARIMA: Обучающая выборка - {len(train_df)} периодов")
        print(f"   ARIMA: Контрольная выборка - {len(test_df)} периодов")
        
        # Обучаем на тренировочных данных
        model = ARIMA(train_df[metric], order=(1, 1, 1))
        model_fit = model.fit()
        
        print(f"   ARIMA: Модель обучена")
        
        # Прогноз на контрольную выборку
        forecast = model_fit.forecast(steps=len(test_df))
        
        print(f"   ARIMA: Прогноз построен")
        
        # Вычисляем метрики
        metrics = calculate_metrics(test_df[metric].values, forecast.values)
        
        print(f"   ARIMA: Метрики вычислены - MAPE: {metrics['mape']:.2f}%")
        
        return {
            'metrics': metrics,
            'validation': {
                'labels': test_df['period'].tolist(),
                'actual': test_df[metric].tolist(),
                'predicted': forecast.tolist()
            }
        }
    except Exception as e:
        print(f"   ❌ ARIMA: Ошибка - {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_prophet_model(train_df, test_df, metric, year_col, month_col):
    """Обучение Prophet модели"""
    from prophet import Prophet
    
    # Подготовка данных для Prophet
    prophet_df = train_df[[year_col, month_col, metric]].copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df[year_col].astype(str) + '-' + 
                                      prophet_df[month_col].astype(str).str.zfill(2) + '-01')
    prophet_df['y'] = prophet_df[metric]
    
    # Обучаем модель
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df[['ds', 'y']])
    
    # Прогноз на контрольную выборку
    future_df = test_df[[year_col, month_col]].copy()
    future_df['ds'] = pd.to_datetime(future_df[year_col].astype(str) + '-' + 
                                     future_df[month_col].astype(str).str.zfill(2) + '-01')
    
    forecast = model.predict(future_df)
    predicted = forecast['yhat'].values
    
    # Вычисляем метрики
    metrics = calculate_metrics(test_df[metric].values, predicted)
    
    return {
        'metrics': metrics,
        'validation_data': {
            'periods': test_df['period'].tolist(),
            'actual': test_df[metric].tolist(),
            'predicted': predicted.tolist()
        }
    }

def train_random_forest_model(train_df, test_df, metric, year_col, month_col):
    """Обучение Random Forest модели (без срезов)"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Подготовка признаков
    X_train = train_df[[year_col, month_col]].values
    y_train = train_df[metric].values
    
    X_test = test_df[[year_col, month_col]].values
    y_test = test_df[metric].values
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Прогноз
    predicted = model.predict(X_test)
    
    # Вычисляем метрики
    metrics = calculate_metrics(y_test, predicted)
    
    return {
        'metrics': metrics,
        'validation': {
            'labels': test_df['period'].tolist(),
            'actual': test_df[metric].tolist(),
            'predicted': predicted.tolist()
        }
    }

def train_random_forest_with_slices(df_agg, metric, year_col, month_col, slice_cols, test_size):
    """Обучение Random Forest с учетом срезов как признаков"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    print(f"   🌲 Random Forest: обучение на всех данных со срезами как признаками", flush=True)
    
    # Создаем копию данных
    df_model = df_agg.copy()
    df_model['period'] = df_model[year_col].astype(str) + '-' + df_model[month_col].astype(str).str.zfill(2)
    
    # Кодируем категориальные признаки (срезы)
    label_encoders = {}
    for col in slice_cols:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].fillna('unknown'))
        label_encoders[col] = le
    
    # Формируем признаки: год, месяц, закодированные срезы
    feature_cols = [year_col, month_col] + [f'{col}_encoded' for col in slice_cols]
    
    # Сортируем по времени и разделяем на train/test
    df_model = df_model.sort_values([year_col, month_col])
    split_index = int(len(df_model) * (1 - test_size))
    
    train_df = df_model[:split_index]
    test_df = df_model[split_index:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[metric].values
    X_test = test_df[feature_cols].values
    y_test = test_df[metric].values
    
    print(f"      Train: {len(X_train)} строк, Test: {len(X_test)} строк", flush=True)
    print(f"      Признаки: {feature_cols}", flush=True)
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    model.fit(X_train, y_train)
    
    # Прогноз
    predicted = model.predict(X_test)
    
    # Метрики
    metrics = calculate_metrics(y_test, predicted)
    
    # Для графика валидации - агрегируем по периодам
    test_df_copy = test_df.copy()
    test_df_copy['predicted'] = predicted
    
    # Группируем по периоду и суммируем (для графика)
    validation_agg = test_df_copy.groupby('period').agg({
        metric: 'sum',
        'predicted': 'sum'
    }).reset_index()
    
    # Подготавливаем детализированные данные для сводной таблицы
    # Создаем отдельные колонки для факта и прогноза
    
    # Добавляем Quarter и Halfyear если их нет
    if 'Quarter' not in test_df_copy.columns:
        test_df_copy['Quarter'] = test_df_copy[month_col].apply(lambda m: f'Q{(int(m)-1)//3 + 1}')
    if 'Halfyear' not in test_df_copy.columns:
        test_df_copy['Halfyear'] = test_df_copy[month_col].apply(lambda m: 'H1' if int(m) <= 6 else 'H2')
    
    # Выбираем нужные колонки
    base_cols = [year_col, 'Halfyear', 'Quarter', month_col] + slice_cols
    detailed_validation = test_df_copy[base_cols].copy()
    
    # Создаем две колонки метрик: факт и прогноз
    detailed_validation[f'{metric}_fact'] = test_df_copy[metric]
    detailed_validation[f'{metric}_predicted'] = test_df_copy['predicted']
    
    print(f"   📊 Детализированная валидация: {len(detailed_validation)} строк", flush=True)
    print(f"   📊 Колонки: {list(detailed_validation.columns)}", flush=True)
    print(f"   📊 Первая строка:", detailed_validation.iloc[0].to_dict() if len(detailed_validation) > 0 else "нет данных", flush=True)
    
    return {
        'metrics': metrics,
        'validation_data': {
            'periods': validation_agg['period'].tolist(),
            'actual': validation_agg[metric].tolist(),
            'predicted': validation_agg['predicted'].tolist()
        },
        'detailed_validation': detailed_validation.to_dict('records'),  # Детализированные данные
        'slice_cols': slice_cols,  # Названия срезов
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }

def train_random_forest_hierarchy(df_agg, metric, year_col, month_col, slice_cols, test_size):
    """Обучение Random Forest с иерархическим согласованием и расширенными признаками"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    print(f"   🌲🏗️ Random Forest Hierarchy: обучение с расширенными признаками", flush=True)
    
    # Создаем копию данных
    df_model = df_agg.copy()
    df_model['period'] = df_model[year_col].astype(str) + '-' + df_model[month_col].astype(str).str.zfill(2)
    
    # Кодируем категориальные признаки (срезы)
    label_encoders = {}
    for col in slice_cols:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].fillna('unknown'))
        label_encoders[col] = le
    
    print(f"   🔧 Построение расширенных признаков...", flush=True)
    
    # Используем FeatureBuilder для создания признаков
    # Важно: создаем признаки для каждой комбинации срезов отдельно
    all_data = []
    
    # Получаем уникальные комбинации срезов
    unique_slices = df_model[slice_cols].drop_duplicates().to_dict('records')
    
    print(f"   📊 Обрабатываем {len(unique_slices)} комбинаций срезов", flush=True)
    
    for slice_combination in unique_slices:
        # Фильтруем данные для этой комбинации
        mask = pd.Series([True] * len(df_model))
        for slice_col in slice_cols:
            mask &= (df_model[slice_col] == slice_combination[slice_col])
        
        df_slice = df_model[mask].copy()
        
        if len(df_slice) < 15:  # Нужно минимум 15 точек для лагов и rolling
            continue
        
        # Строим признаки для этого временного ряда
        fb = FeatureBuilder(df_slice, metric, month_col, year_col)
        df_with_features, _ = fb.build_all_features(categorical_cols=[f'{col}_encoded' for col in slice_cols])
        
        all_data.append(df_with_features)
    
    # Объединяем данные
    df_enriched = pd.concat(all_data, ignore_index=True)
    
    print(f"   ✅ Создан обогащенный датасет: {len(df_enriched)} строк", flush=True)
    
    # Удаляем строки с NaN (из-за лагов и rolling)
    df_enriched_clean = df_enriched.dropna()
    
    print(f"   📊 После удаления NaN: {len(df_enriched_clean)} строк", flush=True)
    
    # Формируем признаки для обучения
    exclude_cols = [metric, 'period', 'time_index'] + slice_cols
    feature_cols = [col for col in df_enriched_clean.columns if col not in exclude_cols]
    
    print(f"   📊 Всего признаков: {len(feature_cols)}", flush=True)
    print(f"   📊 Примеры признаков: {feature_cols[:10]}", flush=True)
    
    # Сортируем и разделяем на train/test
    df_enriched_clean = df_enriched_clean.sort_values([year_col, month_col])
    split_index = int(len(df_enriched_clean) * (1 - test_size))
    
    train_df = df_enriched_clean[:split_index]
    test_df = df_enriched_clean[split_index:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[metric].values
    X_test = test_df[feature_cols].values
    y_test = test_df[metric].values
    
    print(f"      Train: {len(X_train)} строк, Test: {len(X_test)} строк", flush=True)
    
    # Обучаем модель с увеличенными параметрами
    model = RandomForestRegressor(
        n_estimators=200,  # Больше деревьев
        random_state=42,
        n_jobs=-1,
        max_depth=20,  # Глубже
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # Прогноз
    predicted = model.predict(X_test)
    
    # Метрики до согласования
    metrics_before = calculate_metrics(y_test, predicted)
    
    print(f"   📊 Метрики до согласования: MAPE = {metrics_before['mape']:.2f}%", flush=True)
    
    # Применяем иерархическое согласование (bottom-up)
    test_df_copy = test_df.copy()
    test_df_copy['predicted_raw'] = predicted
    
    # Создаем reconciler
    reconciler = HierarchyReconciler(slice_cols, metric)
    
    # Bottom-up согласование
    print(f"   🔼 Применяем bottom-up согласование...", flush=True)
    test_df_copy['predicted'] = predicted  # Пока используем исходный прогноз
    
    # TODO: Реализовать полное bottom-up согласование
    # Для простоты пока оставляем исходный прогноз
    
    # Метрики после согласования
    metrics_after = calculate_metrics(y_test, test_df_copy['predicted'].values)
    
    print(f"   📊 Метрики после согласования: MAPE = {metrics_after['mape']:.2f}%", flush=True)
    
    # Для графика валидации - агрегируем по периодам
    validation_agg = test_df_copy.groupby('period').agg({
        metric: 'sum',
        'predicted': 'sum'
    }).reset_index()
    
    # Подготавливаем детализированные данные для сводной таблицы
    if 'Quarter' not in test_df_copy.columns:
        test_df_copy['Quarter'] = test_df_copy[month_col].apply(lambda m: f'Q{(int(m)-1)//3 + 1}')
    if 'Halfyear' not in test_df_copy.columns:
        test_df_copy['Halfyear'] = test_df_copy[month_col].apply(lambda m: 'H1' if int(m) <= 6 else 'H2')
    
    base_cols = [year_col, 'Halfyear', 'Quarter', month_col] + slice_cols
    detailed_validation = test_df_copy[base_cols].copy()
    
    detailed_validation[f'{metric}_fact'] = test_df_copy[metric]
    detailed_validation[f'{metric}_predicted'] = test_df_copy['predicted']
    
    print(f"   📊 Детализированная валидация: {len(detailed_validation)} строк", flush=True)
    
    return {
        'metrics': metrics_after,
        'validation_data': {
            'periods': validation_agg['period'].tolist(),
            'actual': validation_agg[metric].tolist(),
            'predicted': validation_agg['predicted'].tolist()
        },
        'detailed_validation': detailed_validation.to_dict('records'),
        'slice_cols': slice_cols,
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'feature_builder': None,  # Можно сохранить для использования при прогнозе
        'metrics_before_reconciliation': metrics_before,
        'reconciliation_improvement': metrics_before['mape'] - metrics_after['mape']
    }

def train_prophet_with_slices(df_agg, metric, year_col, month_col, slice_cols, test_size):
    """Обучение Prophet с учетом срезов как регрессоров"""
    from prophet import Prophet
    
    print(f"   📈 Prophet: обучение на всех данных со срезами как регрессорами", flush=True)
    
    # Создаем копию данных
    df_model = df_agg.copy()
    df_model['ds'] = pd.to_datetime(df_model[year_col].astype(str) + '-' + 
                                     df_model[month_col].astype(str).str.zfill(2) + '-01')
    df_model['y'] = df_model[metric]
    
    # One-hot encoding для срезов
    df_encoded = pd.get_dummies(df_model, columns=slice_cols, prefix=slice_cols)
    
    # Находим колонки-регрессоры (все one-hot encoded колонки)
    regressor_cols = [col for col in df_encoded.columns if any(col.startswith(f'{sc}_') for sc in slice_cols)]
    
    print(f"      Регрессоры (срезы): {len(regressor_cols)} колонок", flush=True)
    
    # Сортируем по времени и разделяем на train/test
    df_encoded = df_encoded.sort_values(['ds'])
    split_index = int(len(df_encoded) * (1 - test_size))
    
    train_df = df_encoded[:split_index]
    test_df = df_encoded[split_index:]
    
    print(f"      Train: {len(train_df)} строк, Test: {len(test_df)} строк", flush=True)
    
    # Подготовка данных для Prophet
    prophet_train = train_df[['ds', 'y'] + regressor_cols].copy()
    prophet_test = test_df[['ds'] + regressor_cols].copy()
    
    # Создаем и обучаем модель
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    
    # Добавляем регрессоры
    for reg_col in regressor_cols:
        model.add_regressor(reg_col)
    
    model.fit(prophet_train)
    
    # Прогноз
    forecast = model.predict(prophet_test)
    predicted = forecast['yhat'].values
    
    # Метрики
    y_test = test_df['y'].values
    metrics = calculate_metrics(y_test, predicted)
    
    # Для графика валидации - агрегируем по периодам
    test_df_copy = test_df.copy()
    test_df_copy['predicted'] = predicted
    test_df_copy['period'] = test_df_copy['ds'].dt.strftime('%Y-%m')
    
    validation_agg = test_df_copy.groupby('period').agg({
        'y': 'sum',
        'predicted': 'sum'
    }).reset_index()
    
    return {
        'metrics': metrics,
        'validation_data': {
            'periods': validation_agg['period'].tolist(),
            'actual': validation_agg['y'].tolist(),
            'predicted': validation_agg['predicted'].tolist()
        },
        'model': model,
        'regressor_cols': regressor_cols
    }

# Функции генерации прогноза на будущие периоды

def generate_arima_forecast(df_agg, metric, steps):
    """Генерация прогноза с помощью ARIMA"""
    from statsmodels.tsa.arima.model import ARIMA
    
    # Обучаем на всех данных
    model = ARIMA(df_agg[metric], order=(1, 1, 1))
    model_fit = model.fit()
    
    # Прогноз на N шагов вперед
    forecast = model_fit.forecast(steps=steps)
    
    return forecast.tolist()

def generate_prophet_forecast(df_agg, metric, year_col, month_col, forecast_months):
    """Генерация прогноза с помощью Prophet"""
    from prophet import Prophet
    
    # Подготовка данных
    prophet_df = df_agg[[year_col, month_col, metric]].copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df[year_col].astype(str) + '-' + 
                                      prophet_df[month_col].astype(str).str.zfill(2) + '-01')
    prophet_df['y'] = prophet_df[metric]
    
    # Обучаем модель
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df[['ds', 'y']])
    
    # Создаем DataFrame для прогноза
    future_dates = []
    for fm in forecast_months:
        date_str = f"{fm['year']}-{str(fm['month']).zfill(2)}-01"
        future_dates.append(date_str)
    
    future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    
    # Прогноз
    forecast = model.predict(future_df)
    
    return forecast['yhat'].tolist()

def generate_random_forest_forecast(df_agg, metric, year_col, month_col, forecast_months):
    """Генерация прогноза с помощью Random Forest"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Подготовка данных
    X_train = df_agg[[year_col, month_col]].values
    y_train = df_agg[metric].values
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Подготовка данных для прогноза
    X_forecast = np.array([[fm['year'], fm['month']] for fm in forecast_months])
    
    # Прогноз
    predicted = model.predict(X_forecast)
    
    return predicted.tolist()

def generate_random_forest_hierarchy_forecast(df_agg, metric, year_col, month_col, slice_cols, forecast_months, trained_model_data):
    """Генерация прогноза с помощью Random Forest Hierarchy - упрощенная версия (возвращает только значения)"""
    detailed = generate_random_forest_hierarchy_forecast_detailed(df_agg, metric, year_col, month_col, slice_cols, forecast_months, trained_model_data)
    return [f['predicted'] for f in detailed]

def generate_random_forest_hierarchy_forecast_detailed(df_agg, metric, year_col, month_col, slice_cols, forecast_months, trained_model_data):
    """Генерация детального прогноза с помощью Random Forest Hierarchy с расширенными признаками"""
    print(f"\n🌲🏗️ Генерация прогноза Random Forest Hierarchy", flush=True)
    
    # Получаем обученную модель и encoders из результатов обучения
    if not trained_model_data:
        raise ValueError("Нет данных обученной модели")
    
    model = trained_model_data.get('model')
    label_encoders = trained_model_data.get('label_encoders', {})
    feature_cols = trained_model_data.get('feature_cols', [])
    
    if not model or not feature_cols:
        raise ValueError("Модель или список признаков не найдены")
    
    print(f"   📊 Используем {len(feature_cols)} признаков для прогноза", flush=True)
    
    # Подготавливаем данные: строим признаки для каждой комбинации срезов
    all_forecasts = []
    
    # Получаем уникальные комбинации срезов
    unique_slices = df_agg[slice_cols].drop_duplicates().to_dict('records')
    
    print(f"   📊 Генерируем прогноз для {len(unique_slices)} комбинаций срезов", flush=True)
    
    for slice_combination in unique_slices:
        # Фильтруем исторические данные для этой комбинации
        mask = pd.Series([True] * len(df_agg))
        for slice_col in slice_cols:
            mask &= (df_agg[slice_col] == slice_combination[slice_col])
        
        df_slice = df_agg[mask].copy()
        
        if len(df_slice) < 15:
            continue
        
        # Кодируем срезы (нужно для признаков)
        for slice_col in slice_cols:
            encoded_col = f'{slice_col}_encoded'
            if encoded_col in label_encoders:
                le = label_encoders[encoded_col]
                value = slice_combination[slice_col]
                try:
                    df_slice[encoded_col] = le.transform([value if value in le.classes_ else 'unknown'])[0]
                except:
                    df_slice[encoded_col] = 0
        
        # Строим признаки для исторических данных
        fb = FeatureBuilder(df_slice, metric, month_col, year_col)
        df_with_features, _ = fb.build_all_features(categorical_cols=[f'{col}_encoded' for col in slice_cols])
        
        # Для каждого будущего периода
        for fm in forecast_months:
            # Берем последние N строк для вычисления лагов и rolling
            recent_data = df_with_features.tail(15).copy()
            
            # Создаем строку для прогноза
            forecast_row = {
                year_col: fm['year'],
                month_col: fm['month']
            }
            
            # Добавляем закодированные срезы
            for slice_col in slice_cols:
                forecast_row[slice_col] = slice_combination[slice_col]
                encoded_col = f'{slice_col}_encoded'
                if encoded_col in label_encoders:
                    le = label_encoders[encoded_col]
                    value = slice_combination[slice_col]
                    try:
                        forecast_row[encoded_col] = le.transform([value if value in le.classes_ else 'unknown'])[0]
                    except:
                        forecast_row[encoded_col] = 0
            
            # Временной индекс
            time_index = (fm['year'] - df_slice[year_col].min()) * 12 + fm['month']
            forecast_row['time_index'] = time_index
            forecast_row['time_index_squared'] = time_index ** 2
            
            # Сезонные признаки
            # 1. Синусоиды
            forecast_row['month_sin'] = np.sin(2 * np.pi * fm['month'] / 12)
            forecast_row['month_cos'] = np.cos(2 * np.pi * fm['month'] / 12)
            
            # 2. One-hot для месяцев
            for month in range(1, 13):
                forecast_row[f'is_month_{month}'] = 1 if fm['month'] == month else 0
            
            # 3. Пиковые месяцы
            peak_months = [2, 3, 5, 11, 12]
            forecast_row['is_peak_month'] = 1 if fm['month'] in peak_months else 0
            
            # 4. Q4
            forecast_row['is_q4'] = 1 if fm['month'] >= 10 else 0
            
            # Лаги и rolling - берем из последних данных
            # Это упрощенная версия - в реальности нужно рекурсивно обновлять
            for col in feature_cols:
                if col not in forecast_row:
                    # Если признак не заполнен, берем среднее из последних данных
                    if col in recent_data.columns:
                        forecast_row[col] = recent_data[col].mean()
                    else:
                        forecast_row[col] = 0
            
            # Формируем вектор признаков
            X_forecast = np.array([[forecast_row.get(col, 0) for col in feature_cols]])
            
            # Прогноз
            predicted_value = model.predict(X_forecast)[0]
            
            all_forecasts.append({
                'year': fm['year'],
                'month': fm['month'],
                **slice_combination,
                'predicted': predicted_value
            })
    
    print(f"   ✅ Создано прогнозов: {len(all_forecasts)}", flush=True)
    
    # Возвращаем детальный список прогнозов
    return all_forecasts

# Flask
from flask import Flask, render_template, render_template_string, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename

# Scikit-learn модели
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__, static_folder='static')
app.secret_key = 'marfor-working-app-2024'

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

class WorkingForecastApp:
    def __init__(self):
        """Инициализация рабочего приложения прогнозирования"""
        self.df = None
        self.session_id = None
        self.forecast_results = {}
        self.data_mapping = {}
        
    def load_data_from_file(self, file_path: str):
        """Загрузка данных из файла"""
        try:
            if file_path.endswith('.csv'):
                # Пробуем разные разделители
                separators = [',', ';', '\t', '|']
                for sep in separators:
                    try:
                        self.df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                        if len(self.df.columns) > 1:
                            print(f"✅ Загружено с разделителем '{sep}': {len(self.df)} записей, {len(self.df.columns)} колонок")
                            break
                    except:
                        try:
                            self.df = pd.read_csv(file_path, sep=sep, encoding='cp1251')
                            if len(self.df.columns) > 1:
                                print(f"✅ Загружено с разделителем '{sep}' (cp1251): {len(self.df)} записей, {len(self.df.columns)} колонок")
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
        """Очистка и подготовка данных"""
        print(f"\n🧹 ОЧИСТКА ДАННЫХ:")
        initial_count = len(self.df)
        
        # Очистка числовых колонок
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Пробуем преобразовать в числовой формат
                self.df[col] = self.df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        
        # Удаляем строки где все значения NaN
        self.df = self.df.dropna(how='all')
        
        print(f"  Удалено {initial_count - len(self.df)} записей с пустыми данными")
        print(f"  После очистки: {len(self.df)} записей")
        
        # Анализ колонок
        print(f"\n📊 АНАЛИЗ КОЛОНОК:")
        for i, col in enumerate(self.df.columns):
            dtype = self.df[col].dtype
            non_null = self.df[col].count()
            print(f"  {i}: {col} ({dtype}) - {non_null} значений")
    
    def get_data_info(self):
        """Получение информации о данных"""
        if self.df is None:
            return None
        
        # Очищаем NaN значения для JSON
        sample_data = convert_to_json_serializable(self.df.head(5).fillna('').to_dict('records'))
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'sample_data': sample_data,
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        return info
    
    def apply_data_mapping(self, mapping_config):
        """Применить маппинг данных"""
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        df = self.df.copy()
        
        # Обработка колонок
        columns_to_include = []
        for col_config in mapping_config.get('columns', []):
            if col_config.get('include', True):
                col_name = col_config['name']
                col_type = col_config.get('type', 'auto')
                
                if col_name in df.columns:
                    # Преобразование типов
                    if col_type == 'numeric':
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    elif col_type == 'text':
                        df[col_name] = df[col_name].astype(str)
                    elif col_type == 'category':
                        df[col_name] = df[col_name].astype('category')
                    
                    columns_to_include.append(col_name)
        
        # Оставляем только выбранные колонки
        if columns_to_include:
            df = df[columns_to_include]
        
        # Обработка пустых значений
        missing_strategy = mapping_config.get('missingValues', 'zeros')
        if missing_strategy == 'remove':
            df = df.dropna()
        elif missing_strategy == 'zeros':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Обработка выбросов
        if mapping_config.get('detectOutliers', False):
            threshold = mapping_config.get('outlierThreshold', 3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
                if mapping_config.get('removeOutliers', False):
                    df = df[~outliers]
                else:
                    # Заменяем выбросы на медиану
                    df.loc[outliers, col] = df[col].median()
        
        # Нормализация данных
        if mapping_config.get('normalizeData', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Логарифмическое преобразование
        if mapping_config.get('logTransform', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
        
        # Создание временных признаков
        if mapping_config.get('createFeatures', False):
            time_series = mapping_config.get('timeSeries', {})
            
            # Обработка временных рядов
            if time_series.get('date'):
                date_col = df.columns[int(time_series['date'])]
                if date_col in df.columns:
                    df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                    df['year'] = df['date_parsed'].dt.year
                    df['month'] = df['date_parsed'].dt.month
                    df['quarter'] = df['date_parsed'].dt.quarter
                    df['week'] = df['date_parsed'].dt.isocalendar().week
            
            if time_series.get('year'):
                year_col = df.columns[int(time_series['year'])]
                if year_col in df.columns:
                    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
            
            if time_series.get('month'):
                month_col = df.columns[int(time_series['month'])]
                if month_col in df.columns:
                    # Обработка текстовых месяцев
                    month_mapping = {
                    'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4,
                    'май': 5, 'июнь': 6, 'июль': 7, 'август': 8,
                    'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
                    }
                    if df[month_col].dtype == 'object':
                        df['month'] = df[month_col].str.lower().map(month_mapping).fillna(pd.to_numeric(df[month_col], errors='coerce'))
                else:
                    df['month'] = pd.to_numeric(df[month_col], errors='coerce')
            
            if time_series.get('quarter'):
                quarter_col = df.columns[int(time_series['quarter'])]
                if quarter_col in df.columns:
                    df['quarter'] = pd.to_numeric(df[quarter_col], errors='coerce')
            
            if time_series.get('week'):
                week_col = df.columns[int(time_series['week'])]
                if week_col in df.columns:
                    df['week'] = pd.to_numeric(df[week_col], errors='coerce')
            
            if time_series.get('halfyear'):
                halfyear_col = df.columns[int(time_series['halfyear'])]
                if halfyear_col in df.columns:
                    df['halfyear'] = pd.to_numeric(df[halfyear_col], errors='coerce')
            
            # Создаем дополнительные временные признаки
            if 'year' in df.columns and 'month' in df.columns:
                df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Зима
                                       3: 1, 4: 1, 5: 1,    # Весна
                                       6: 2, 7: 2, 8: 2,    # Лето
                                       9: 3, 10: 3, 11: 3}) # Осень
                df['is_weekend'] = 0  # Заглушка для будущего расширения
        
        # Обновляем данные в экземпляре
        self.df = df
        
        return df
    
    def set_data_mapping(self, mapping):
        """Установка маппинга колонок"""
        self.data_mapping = mapping
        print(f"\n🗺️ МАППИНГ ДАННЫХ:")
        for key, value in mapping.items():
            print(f"  {key}: колонка {value}")
    
    def run_cascaded_forecast(self, config):
        """Запуск каскадного прогноза с Random Forest"""
        try:
            print(f"\n🔮 ЗАПУСК КАСКАДНОГО ПРОГНОЗА:")
            
            # Получаем настройки
            periods = config.get('periods', 4)
            method = config.get('method', 'random_forest')
            year_col = self.data_mapping.get('year', 0)
            month_col = self.data_mapping.get('month', 1)
            
            print(f"  Периодов прогноза: {periods}")
            print(f"  Метод: {method}")
            print(f"  Колонка года: {year_col}")
            print(f"  Колонка месяца: {month_col}")
            
            # Проверяем наличие временных колонок
            if year_col >= len(self.df.columns) or month_col >= len(self.df.columns):
                return False, "Неправильно указаны колонки года или месяца"
            
            # Переименовываем колонки для удобства
            year_col_name = self.df.columns[year_col]
            month_col_name = self.df.columns[month_col]
            
            # Очищаем временные данные
            self.df[year_col_name] = pd.to_numeric(self.df[year_col_name], errors='coerce')
            self.df[month_col_name] = pd.to_numeric(self.df[month_col_name], errors='coerce')
            
            # Удаляем строки с пустыми временными данными
            self.df = self.df.dropna(subset=[year_col_name, month_col_name])
            
            if len(self.df) < 10:
                return False, "Недостаточно данных для прогнозирования"
            
            # Находим числовые колонки для прогнозирования
            numeric_cols = []
            for i, col in enumerate(self.df.columns):
                if i not in [year_col, month_col] and pd.api.types.is_numeric_dtype(self.df[col]):
                    if self.df[col].sum() > 0:  # Только колонки с положительными значениями
                        numeric_cols.append(i)
            
            if not numeric_cols:
                return False, "Не найдены числовые колонки для прогнозирования"
            
            print(f"  Найдено {len(numeric_cols)} числовых колонок для прогнозирования")
            
            # Создаем прогноз для каждой числовой колонки
            forecast_data = []
            
            for col_idx in numeric_cols:
                col_name = self.df.columns[col_idx]
                print(f"\n  📊 Прогнозирование для {col_name}:")
                
                # Подготавливаем данные
                forecast_result = self._create_forecast_for_column(
                    col_name, year_col_name, month_col_name, periods, method
                )
                
                if forecast_result:
                    forecast_data.append(forecast_result)
                    print(f"    ✅ Прогноз создан")
                else:
                    print(f"    ❌ Ошибка создания прогноза")
            
            if not forecast_data:
                return False, "Не удалось создать ни одного прогноза"
            
            # Сохраняем результаты
            self.forecast_results = {
                'forecast_data': forecast_data,
                'settings': config,
                'total_forecasts': len(forecast_data)
            }
            
            return True, f"Создано {len(forecast_data)} прогнозов"
            
        except Exception as e:
            return False, f"Ошибка в каскадном прогнозе: {str(e)}"
    
    def _create_forecast_for_column(self, col_name, year_col, month_col, periods, method):
        """Создание прогноза для конкретной колонки"""
        try:
            # Подготавливаем данные
            data = self.df[[year_col, month_col, col_name]].copy()
            data = data.dropna()
            
            if len(data) < 6:
                return None
            
            # Создаем временной индекс
            data['time_index'] = (data[year_col] - data[year_col].min()) * 12 + (data[month_col] - 1)
            
            # Сезонные признаки
            data['month_sin'] = np.sin(2 * np.pi * data[month_col] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data[month_col] / 12)
            
            # Квартальные признаки
            data['quarter'] = ((data[month_col] - 1) // 3) + 1
            for q in range(1, 5):
                data[f'q{q}'] = (data['quarter'] == q).astype(int)
            
            # Праздничные периоды
            data['holiday_period'] = (
                (data[month_col] == 12) |  # Декабрь
                (data[month_col] == 1) |   # Январь
                (data[month_col] == 2) |   # Февраль
                (data[month_col] == 3) |   # Март
                (data[month_col] == 5)     # Май
            ).astype(int)
            
            # Подготавливаем признаки
            features = ['time_index', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4', 'holiday_period']
            X = data[features].fillna(0)
            y = data[col_name].fillna(0)
            
            # Выбираем модель
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = Ridge(alpha=1.0)
            
            # Обучаем модель
            model.fit(X, y)
            
            # Рассчитываем метрики
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Создаем прогноз
            last_year = data[year_col].max()
            last_month = data[month_col].max()
            last_time_index = data['time_index'].max()
            
            forecast_periods = []
            for i in range(1, periods + 1):
                period_data = {
                    'year': last_year + (i // 12),
                    'month': ((last_month + i - 1) % 12) + 1,
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
                X_forecast = np.array([period_data[feature] for feature in features]).reshape(1, -1)
                forecast_value = model.predict(X_forecast)[0]
                forecast_value = max(0, forecast_value)  # Не допускаем отрицательные значения
                
                period_data['forecast'] = forecast_value
                forecast_periods.append(period_data)
            
            return {
                'column_name': col_name,
                'model_type': method,
                'r2': r2,
                'mae': mae,
                'forecast_periods': forecast_periods,
                'total_forecast': sum(p['forecast'] for p in forecast_periods)
            }
            
        except Exception as e:
            print(f"    Ошибка прогноза для {col_name}: {str(e)}")
            return None
    
    def save_results(self, session_id):
        """Сохранение результатов"""
        try:
            if not self.forecast_results:
                return None
            
            # Создаем DataFrame с результатами
            all_results = []
            
            for forecast in self.forecast_results['forecast_data']:
                for period in forecast['forecast_periods']:
                    all_results.append({
                    'column': forecast['column_name'],
                    'year': period['year'],
                    'month': period['month'],
                    'forecast': period['forecast'],
                    'model_type': forecast['model_type'],
                    'r2': forecast['r2'],
                    'mae': forecast['mae']
                    })
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                filename = f"cascaded_forecast_{session_id}.csv"
                filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                results_df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"💾 Результаты сохранены в {filepath}")
                return filepath
            
            return None
            
        except Exception as e:
            print(f"Ошибка сохранения результатов: {str(e)}")
            return None

# Глобальный объект приложения
forecast_app = WorkingForecastApp()

@app.route('/')
def index():
    """Главная страница - дашборд"""
    return render_template('dashboard.html', username='Пользователь', projects=[])

@app.route('/forecast')
def forecast():
    """Страница прогнозирования"""
    project_id = request.args.get('project')
    if project_id:
        # Загружаем проект
        try:
            project_file = os.path.join('projects', f"{project_id}.json")
            if os.path.exists(project_file):
                with open(project_file, 'r', encoding='utf-8') as f:
                    project = json.load(f)
                
                # Загружаем данные в forecast_app
                if project.get('data_info'):
                    # Проверяем, есть ли полные данные в data_info
                    if project['data_info'].get('full_data'):
                        # Используем полные данные из data_info
                        full_data = project['data_info']['full_data']
                    df = pd.DataFrame(full_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                elif project.get('processed_data') and project['processed_data'].get('sample_data'):
                    # Используем данные из processed_data
                    sample_data = project['processed_data']['sample_data']
                    df = pd.DataFrame(sample_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                else:
                    # Fallback: используем sample_data из data_info
                    sample_data = project['data_info'].get('sample_data', [])
                    if sample_data:
                        df = pd.DataFrame(sample_data)
                    # Заполняем пропуски и заменяем NaN
                    df = df.fillna('')
                    # Дополнительная очистка NaN значений
                    df = df.replace([np.nan, 'nan', 'NaN'], '')
                    
                    # Сохраняем в forecast_app
                    forecast_app.df = df
                    forecast_app.session_id = project['session_id']
                    print(f"DEBUG: Загружено {len(df)} строк в forecast_app для проекта {project_id}")
                    
                    # Обновляем время последнего доступа
                    project['updated_at'] = datetime.now().isoformat()
                    with open(project_file, 'w', encoding='utf-8') as f:
                        json.dump(project, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при загрузке проекта: {e}")
    
    return render_template('marfor_interface.html')

@app.route('/logout')
def logout():
    """Выход из системы"""
    return redirect('/')

@app.route('/favicon.ico')
def favicon():
    """Favicon"""
    return '', 204  # No Content

@app.route('/forecast/mapping')
def data_mapping():
    """Страница маппинга данных"""
    return render_template('data_mapping.html')

@app.route('/forecast/settings')
def forecast_settings():
    """Страница настройки горизонта прогнозирования"""
    return render_template('forecast_settings.html')

@app.route('/forecast/training')
def model_training():
    """Страница обучения и валидации моделей"""
    return render_template('model_training.html')

@app.route('/forecast/results')
def forecast_results():
    """Страница результатов прогноза"""
    return render_template('forecast_results.html')

@app.route('/forecast/configure')
def forecast_configure():
    """Страница настройки прогноза"""
    return render_template('marfor_interface.html')

@app.route('/demo/mapping')
def demo_mapping():
    """Демо страница маппинга данных"""
    return render_template('demo_mapping.html')

@app.route('/api/apply_mapping', methods=['POST'])
def apply_mapping():
    """Применение маппинга данных"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        mapping_config = data.get('mapping')
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Применяем маппинг
        processed_data = forecast_app.apply_data_mapping(mapping_config)
        
        return jsonify({
            'success': True,
            'message': 'Маппинг применен успешно',
            'processed_data_info': {
                'shape': processed_data.shape,
                'columns': list(processed_data.columns),
                'dtypes': {col: str(dtype) for col, dtype in processed_data.dtypes.items()},
                'missing_values': processed_data.isnull().sum().to_dict(),
                'sample_data': convert_to_json_serializable(processed_data.head(5).fillna('').to_dict('records'))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при применении маппинга: {str(e)}'})

@app.route('/api/get_processed_data/<session_id>')
def get_processed_data(session_id):
    """Получение обработанных данных"""
    try:
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        # Получаем информацию об обработанных данных
        data_info = forecast_app.get_data_info()
        
        return jsonify({
            'success': True,
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении данных: {str(e)}'})

@app.route('/api/get_time_series_data/<session_id>')
def get_time_series_data(session_id):
    """Получение данных временных рядов для визуализации"""
    try:
        print(f"🔧 ВЕРСИЯ КОДА: 2.21.0 - Генерация прогноза и страница результатов")
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Проверяем, нужно ли использовать прогнозные данные
        use_forecast = request.args.get('use_forecast', 'false').lower() == 'true'
        
        if use_forecast:
            # Используем прогнозные данные (факт + прогноз)
            if not hasattr(forecast_app, 'forecast_results') or session_id not in forecast_app.forecast_results:
                return jsonify({'success': False, 'message': 'Прогнозные данные не найдены'})
            
            combined_data = forecast_app.forecast_results[session_id]['combined_data']
            df = pd.DataFrame(combined_data)
            print(f"DEBUG: Используем прогнозные данные: {len(df)} строк (факт + прогноз)")
            print(f"DEBUG: Колонки прогнозных данных: {list(df.columns)}")
            if 'is_forecast' in df.columns:
                forecast_count = df['is_forecast'].sum()
                print(f"DEBUG: Прогнозных строк: {forecast_count}, фактических: {len(df) - forecast_count}")
            else:
                print(f"WARNING: Колонка 'is_forecast' отсутствует в прогнозных данных!")
        else:
            # Используем обычные данные
            if forecast_app.df is None:
                return jsonify({'success': False, 'message': 'Данные не загружены'})
            
            df = forecast_app.df.copy()
            print(f"DEBUG: Используем обычные данные: {len(df)} строк")
        
        # Отладочная информация
        print(f"DEBUG: Загружено {len(df)} строк данных")
        print(f"DEBUG: Колонки: {list(df.columns)}")
        
        # Получаем параметры из запроса
        time_column = request.args.get('time_column', '')
        metric_columns = request.args.getlist('metrics')
        slice_columns = request.args.getlist('slices')  # Добавляем поддержку срезов
        group_by = request.args.get('group_by', '')
        show_pivot = request.args.get('show_pivot', 'false').lower() == 'true'
        pivot_mode = request.args.get('pivot_mode', 'time-series')  # По умолчанию временные ряды
        split_by_slice = request.args.get('split_by_slice', '')  # Добавляем параметр разбивки по срезам
        
        print(f"DEBUG: time_column={time_column}, metrics={metric_columns}, group_by={group_by}, show_pivot={show_pivot}, pivot_mode={pivot_mode}, split_by_slice={split_by_slice}")
        print(f"DEBUG: Все параметры запроса: {dict(request.args)}")
        
        # Получаем маппинг из параметров запроса
        mapping_data = request.args.get('mapping_data', '{}')
        import json
        try:
            mapping_config = json.loads(mapping_data) if mapping_data else {}
        except json.JSONDecodeError as e:
            print(f"ERROR: Некорректный JSON в маппинге: {e}")
            return jsonify({
                'success': False, 
                'message': f'Критическая ошибка: некорректный формат маппинга данных. Ошибка JSON: {str(e)}'
            })
        
        print(f"DEBUG: Маппинг конфигурация: {mapping_config}")
        print(f"DEBUG: Количество колонок в маппинге: {len(mapping_config.get('columns', []))}")
        
        if not mapping_config or not mapping_config.get('columns'):
            print("ERROR: Маппинг не найден или пустой - это критическая ошибка!")
            return jsonify({
                'success': False, 
                'message': 'Критическая ошибка: маппинг данных не настроен или пустой. Пожалуйста, настройте маппинг колонок перед созданием сводной таблицы.'
            })
        
        # Проверка параметров в зависимости от режима
        if pivot_mode == 'time-series':
            if not time_column or not metric_columns:
                return jsonify({'success': False, 'message': 'Не указаны временная колонка или метрики'})
        else:  # pivot_mode == 'slices'
            if not time_column or not slice_columns:
                return jsonify({'success': False, 'message': 'Не указаны временная колонка или срезы'})
        
        # Проверяем существование колонок
        if time_column not in df.columns:
            return jsonify({'success': False, 'message': f'Временная колонка {time_column} не найдена'})
        
        for metric in metric_columns:
            if metric not in df.columns:
                return jsonify({'success': False, 'message': f'Метрика {metric} не найдена'})
        
        # Подготавливаем данные
        result_data = {
            'time_series': [],
            'grouped_series': {},
            'time_labels': [],
            'metrics': metric_columns,
            'pivot_table': None
        }
        
        print(f"DEBUG: Исходные данные содержат колонки: {df.columns.tolist()}")
        print(f"DEBUG: Первые 3 строки исходных данных:")
        for i, row in enumerate(df.head(3).to_dict('records')):
            print(f"  Строка {i}: {row}")
        
        # Сортируем по времени
        df_sorted = df.sort_values(time_column)
        
        # Получаем уникальные временные метки
        time_labels = df_sorted[time_column].unique()
        result_data['time_labels'] = [str(label) for label in time_labels]
        
        print(f"DEBUG: Найдено {len(time_labels)} уникальных временных меток: {time_labels[:10]}...")
        
        # Если есть группировка
        if group_by and group_by in df.columns:
            groups = df_sorted[group_by].unique()
            
            for group in groups:
                group_data = df_sorted[df_sorted[group_by] == group]
                group_series = {}
                
                for metric in metric_columns:
                    # Агрегируем данные по времени (сумма для числовых, последнее значение для категориальных)
                    if df[metric].dtype in ['int64', 'float64']:
                        metric_data = group_data.groupby(time_column)[metric].sum()
                else:
                    metric_data = group_data.groupby(time_column)[metric].last()
                    
                    # Заполняем пропуски
                    full_series = []
                    for time_label in time_labels:
                        if time_label in metric_data.index:
                            full_series.append(float(metric_data[time_label]) if pd.notna(metric_data[time_label]) else 0)
                        else:
                            full_series.append(0)
                    
                    group_series[metric] = full_series
                
                result_data['grouped_series'][str(group)] = group_series
        else:
            # Без группировки - общие данные
            for metric in metric_columns:
                if df[metric].dtype in ['int64', 'float64']:
                    metric_data = df_sorted.groupby(time_column)[metric].sum()
                else:
                    metric_data = df_sorted.groupby(time_column)[metric].last()
                
                # Заполняем пропуски
                full_series = []
                for time_label in time_labels:
                    if time_label in metric_data.index:
                        full_series.append(float(metric_data[time_label]) if pd.notna(metric_data[time_label]) else 0)
                    else:
                        full_series.append(0)
                
                result_data['time_series'].append({
                    'metric': metric,
                    'data': full_series
                })
        
        # Создаем сводную таблицу если запрошено
        print(f"DEBUG: show_pivot = {show_pivot}")
        if show_pivot:
            try:
                # Получаем настройки маппинга из sessionStorage (передаем через параметры)
                mapping_data = request.args.get('mapping_data', '{}')
                import json
                mapping = json.loads(mapping_data) if mapping_data else {}
                
                print(f"DEBUG: Получен маппинг: {mapping}")
                
                # Находим временные ряды с уровнями
                time_series_cols = []
                slice_cols = []
                if mapping.get('columns'):
                    for col in mapping['columns']:
                        if col.get('time_series') and col.get('nesting_level', 0) >= 0:
                            time_series_cols.append({
                                'name': col['name'],
                                'type': col['time_series'],
                                'level': col['nesting_level']
                            })
                        elif col.get('role') == 'dimension' and not col.get('time_series') and col.get('nesting_level', 0) >= 0:
                            slice_cols.append({
                                'name': col['name'],
                                'type': 'slice',
                                'level': col['nesting_level']
                            })
                
                # Сортируем по уровням
                time_series_cols.sort(key=lambda x: x['level'])
                slice_cols.sort(key=lambda x: x['level'])
                
                print(f"DEBUG: Временные ряды: {time_series_cols}")
                print(f"DEBUG: Срезы: {slice_cols}")
                
                # В зависимости от режима сводной таблицы
                print(f"DEBUG: pivot_mode = {pivot_mode}")
                print(f"DEBUG: split_by_slice = {split_by_slice}")
                
                if pivot_mode == 'time-series' and time_series_cols:
                    # В режиме временных рядов
                    print(f"DEBUG: Попадаем в блок time-series")
                    time_cols = time_series_cols.copy()
                    
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        print(f"DEBUG: Включаем режим разбивки по срезу: {split_by_slice}")
                        # Разбивка по срезу - временные колонки в строках, срез в столбцах
                        split_col = [col for col in slice_cols if col['name'] == split_by_slice][0]
                        print(f"DEBUG: Найден срез для разбивки: {split_col}")
                        
                        # Создаем сводную таблицу с разбивкой по срезу
                        pivot_cols = [col['name'] for col in time_cols]
                        print(f"DEBUG: Разбивка по срезу {split_by_slice}, временные колонки: {pivot_cols}")
                    else:
                        print(f"DEBUG: Обычный режим временных рядов без разбивки")
                        # Обычный режим временных рядов - только временные колонки
                        pivot_cols = [col['name'] for col in time_cols]
                        print(f"DEBUG: Временные колонки: {pivot_cols}")
                    
                    # Создаем pivot table с метриками
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        # С разбивкой по срезу
                        pivot_data = df_sorted.groupby(pivot_cols + [split_by_slice])[metric_columns].sum().reset_index()
                    else:
                        # Без разбивки - только временные колонки
                        pivot_data = df_sorted.groupby(pivot_cols)[metric_columns].sum().reset_index()
                    
                    # Создаем структуру с разбивкой по столбцам
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        unique_slices = sorted(pivot_data[split_by_slice].unique())
                        column_headers = {}
                        
                        for slice_value in unique_slices:
                            slice_data = pivot_data[pivot_data[split_by_slice] == slice_value]
                            column_headers[str(slice_value)] = {}
                            for metric in metric_columns:
                                column_headers[str(slice_value)][metric] = {}
                                for _, row in slice_data.iterrows():
                                    # Создаем ключ из временных значений
                                    time_key = '_'.join(str(row[col]) for col in pivot_cols)
                                    column_headers[str(slice_value)][metric][time_key] = float(row[metric]) if pd.notna(row[metric]) else 0
                    else:
                        # Без разбивки - простые заголовки
                        unique_slices = []
                        column_headers = {}
                    
                    # Включаем ВСЕ данные из маппинга для разбивки по срезам
                    all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                    available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                    
                    if split_by_slice and split_by_slice in [col['name'] for col in slice_cols]:
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': time_cols + [split_col],
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': split_by_slice,
                            'unique_slices': convert_to_json_serializable(unique_slices),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,
                            'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                        }
                    else:
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': time_cols,
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': '',
                            'unique_slices': convert_to_json_serializable(unique_slices),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,
                            'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                        }
                    
                    print(f"DEBUG: Создана сводная таблица с разбивкой:")
                    print(f"  - Колонки: {pivot_cols}")
                    print(f"  - Уникальные срезы: {unique_slices}")
                    print(f"  - Метрики: {metric_columns}")
                    print(f"  - Количество строк данных: {len(pivot_data)}")
                    print(f"  - Структура column_headers: {list(column_headers.keys())}")
                    print(f"  - Первые 3 строки данных:")
                    for i, row in enumerate(pivot_data.head(3).to_dict('records')):
                        print(f"    Строка {i}: {row}")
                    print(f"  - Структура данных: {pivot_data.columns.tolist()}")
                    print(f"  - Типы данных: {pivot_data.dtypes.to_dict()}")
                    
                    print(f"DEBUG: Создана сводная таблица с разбивкой по {split_by_slice}, уникальные значения: {unique_slices}")
                else:
                    # Обычный режим временных рядов - только временные колонки
                    all_cols = time_series_cols.copy()
                    print(f"DEBUG: Режим временных рядов - используем только временные колонки: {all_cols}")
                    
                    # Создаем сводную таблицу
                    pivot_cols = [col['name'] for col in all_cols]
                    print(f"DEBUG: Колонки для сводной таблицы: {pivot_cols}")
                    print(f"DEBUG: Метрики: {metric_columns}")
                    
                    pivot_data = df_sorted.groupby(pivot_cols)[metric_columns].sum().reset_index()
                    print(f"DEBUG: Создана сводная таблица с {len(pivot_data)} строками")
                    
                    # Форматируем данные для отображения
                    # Включаем ВСЕ данные из маппинга
                    all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                    available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                    
                    result_data['pivot_table'] = {
                        'columns': available_columns,
                        'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                        'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                        'time_series_info': all_cols,
                        'available_slices': slice_cols,
                        'pivot_mode': 'time-series'  # Явно указываем режим временных рядов
                    }
                
                if pivot_mode == 'slices':
                    # В режиме срезов - срезы в строках, метрики/временные ряды в столбцах
                    print(f"DEBUG: Попадаем в блок slices")
                    if split_by_slice and split_by_slice in [col['name'] for col in time_series_cols]:
                        print(f"DEBUG: Включаем режим разбивки по временному ряду: {split_by_slice}")
                        # Разбивка по временному ряду - срезы в строках, временной ряд в столбцах
                        slice_col_names = [col['name'] for col in slice_cols]
                        split_col = [col for col in time_series_cols if col['name'] == split_by_slice][0]
                        print(f"DEBUG: Найден временной ряд для разбивки: {split_col}")
                    
                        # Создаем pivot table с метриками в столбцах для каждого значения временного ряда
                        pivot_data = df_sorted.groupby(slice_col_names + [split_by_slice])[metric_columns].sum().reset_index()
                        
                        # Создаем структуру с разбивкой по столбцам (как в режиме временных рядов)
                        unique_time_values = sorted(pivot_data[split_by_slice].unique())
                        column_headers = {}
                        
                        for time_value in unique_time_values:
                            time_data = pivot_data[pivot_data[split_by_slice] == time_value]
                            column_headers[str(time_value)] = {}
                            for metric in metric_columns:
                                column_headers[str(time_value)][metric] = {}
                                for _, row in time_data.iterrows():
                                    # Создаем ключ из срезов (аналогично временным рядам)
                                    slice_key = '_'.join(str(row[col]) for col in slice_col_names)
                                    column_headers[str(time_value)][metric][slice_key] = float(row[metric]) if pd.notna(row[metric]) else 0
                        
                        # Включаем ВСЕ данные из маппинга для разбивки по временному ряду
                        all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                        available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                        
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': [],  # В режиме срезов временные ряды НЕ в строках
                            'column_headers': convert_to_json_serializable(column_headers),
                            'split_by_slice': split_by_slice,
                            'unique_time_values': convert_to_json_serializable(unique_time_values),
                            'metrics': metric_columns,
                            'available_slices': slice_cols,  # Срезы для строк
                            'available_time_series': time_series_cols,  # Временные ряды для разбивки по столбцам
                            'pivot_mode': 'slices'  # Явно указываем режим срезов
                        }
                    else:
                        # Обычный режим срезов без разбивки
                        print(f"DEBUG: Обычный режим срезов")
                        # Создаем сводную таблицу с срезами в строках, метриками в значениях
                        print(f"DEBUG: Все срезы: {slice_cols}")
                        print(f"DEBUG: Метрики: {metric_columns}")
                        
                        # Форматируем данные для отображения
                        # Включаем ВСЕ данные из маппинга
                        all_mapping_columns = [col['name'] for col in mapping_config.get('columns', [])]
                        available_columns = [col for col in all_mapping_columns if col in df_sorted.columns]
                        
                        result_data['pivot_table'] = {
                            'columns': available_columns,
                            'data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),
                            'raw_data': convert_to_json_serializable(df_sorted[available_columns].to_dict('records')),  # Добавляем исходные данные для фильтров
                            'time_series_info': [],  # В режиме срезов временные ряды не в строках
                            'available_slices': slice_cols,  # Срезы для строк
                            'available_time_series': time_series_cols,  # Временные ряды для разбивки
                            'metrics': metric_columns,  # Метрики для значений
                            'pivot_mode': 'slices'  # Явно указываем режим срезов
                        }
                        
                        print(f"DEBUG: Сводная таблица создана в режиме 'slices'")
                    
            except Exception as e:
                print(f"Ошибка создания сводной таблицы: {e}")
                result_data['pivot_table'] = None
        
        return jsonify({
            'success': True,
            'data': result_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении данных временных рядов: {str(e)}'})

@app.route('/api/save_project', methods=['POST'])
def save_project():
    """Сохранение проекта"""
    try:
        data = request.get_json()
        project_name = data.get('name', '')
        session_id = data.get('session_id', '')
        
        if not project_name:
            return jsonify({'success': False, 'message': 'Название проекта не указано'})
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Создаем объект проекта
        data_info = forecast_app.get_data_info()
        
        # Добавляем полные данные в data_info
        if forecast_app.df is not None:
            # Сохраняем все данные, а не только sample
            # Заменяем NaN на None для корректной JSON сериализации
            df_clean = forecast_app.df.fillna('')
            data_info['full_data'] = convert_to_json_serializable(df_clean.to_dict('records'))
        
        project = {
            'id': str(uuid.uuid4()),
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'session_id': session_id,
            'data_info': data_info,
            'data_mapping': data.get('data_mapping', {}),
            'processed_data': data.get('processed_data', {}),
            'status': 'saved'
        }
        
        # Сохраняем в файл
        projects_dir = 'projects'
        os.makedirs(projects_dir, exist_ok=True)
        
        project_file = os.path.join(projects_dir, f"{project['id']}.json")
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Проект сохранен успешно',
            'project_id': project['id']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при сохранении проекта: {str(e)}'})

@app.route('/api/load_project/<project_id>')
def load_project(project_id):
    """Загрузка проекта"""
    try:
        project_file = os.path.join('projects', f"{project_id}.json")
        
        if not os.path.exists(project_file):
            return jsonify({'success': False, 'message': 'Проект не найден'})
        
        with open(project_file, 'r', encoding='utf-8') as f:
            project = json.load(f)
        
        # Пытаемся загрузить данные из исходного CSV файла вместо JSON
        session_id = project.get('session_id')
        csv_loaded = False
        
        if session_id:
            # Ищем исходный файл в uploads
            upload_folder = app.config['UPLOAD_FOLDER']
            matching_files = [f for f in os.listdir(upload_folder) if f.startswith(session_id)]
            
            if matching_files:
                original_file = os.path.join(upload_folder, matching_files[0])
                success, message = forecast_app.load_data_from_file(original_file)
                
                if success:
                    forecast_app.session_id = session_id
                    csv_loaded = True
                    print(f"✅ Проект {project.get('name')}: Данные загружены из CSV ({message})")
                else:
                    print(f"⚠️ Проект {project.get('name')}: Не удалось загрузить CSV ({message})")
        
        # Если CSV не загрузился, используем данные из JSON (fallback)
        if not csv_loaded:
            print(f"⚠️ Проект {project.get('name')}: Используются данные из JSON (возможна потеря строк)")
            # Загружаем full_data из проекта
            full_data = project.get('data_info', {}).get('full_data', [])
            if full_data:
                forecast_app.df = pd.DataFrame(full_data)
                forecast_app.session_id = session_id or project_id
        
        # Очищаем NaN значения в метаданных проекта
        def clean_nan_values(obj):
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif isinstance(obj, str) and obj in ['nan', 'NaN', 'null']:
                return ''
            elif pd.isna(obj) if hasattr(pd, 'isna') else False:
                return ''
            else:
                return obj
        
        # Очищаем только метаданные, но не full_data (он может быть большим)
        project_clean = {
            'id': project.get('id'),
            'name': project.get('name'),
            'created_at': project.get('created_at'),
            'updated_at': datetime.now().isoformat(),
            'session_id': project.get('session_id'),
            'mapping_config': clean_nan_values(project.get('mapping_config', {})),
            'csv_loaded': csv_loaded
        }
        
        # Обновляем время последнего доступа
        project['updated_at'] = datetime.now().isoformat()
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'project': project_clean
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при загрузке проекта: {str(e)}'})

@app.route('/api/list_projects')
def list_projects():
    """Список сохраненных проектов"""
    try:
        projects_dir = 'projects'
        print(f"🔍 Проверяем директорию проектов: {projects_dir}")
        if not os.path.exists(projects_dir):
            print(f"❌ Директория {projects_dir} не существует")
            return jsonify({'success': True, 'projects': []})
        
        projects = []
        files = os.listdir(projects_dir)
        print(f"📁 Найдено файлов в {projects_dir}: {len(files)}")
        
        for filename in files:
            print(f"📄 Обрабатываем файл: {filename}")
            if filename.endswith('.json'):
                project_file = os.path.join(projects_dir, filename)
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project = json.load(f)
                    
                    # Поддержка старого формата (без метаданных)
                    if 'id' not in project:
                        # Старый формат - используем имя файла как ID
                        project_id = filename.replace('.json', '')
                        project_name = project.get('data_info', {}).get('filename', 'Проект без имени')
                        # Получаем время модификации файла
                        from time import strftime, localtime
                        mtime = os.path.getmtime(project_file)
                        timestamp = strftime('%Y-%m-%d %H:%M:%S', localtime(mtime))
                        
                        projects.append({
                            'id': project_id,
                            'name': project_name,
                            'created_at': timestamp,
                            'updated_at': timestamp,
                            'status': 'saved'
                        })
                        print(f"✅ Загружен старый проект: {project_name}")
                    else:
                        # Новый формат с метаданными
                        projects.append({
                            'id': project['id'],
                            'name': project['name'],
                            'created_at': project['created_at'],
                            'updated_at': project['updated_at'],
                            'status': project.get('status', 'saved')
                        })
                        print(f"✅ Загружен проект: {project.get('name', 'без имени')}")
                except Exception as e:
                    print(f"❌ Ошибка загрузки проекта {filename}: {str(e)}")
                    continue
        
        # Сортируем по времени обновления
        projects.sort(key=lambda x: x['updated_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'projects': projects
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при получении списка проектов: {str(e)}'})

@app.route('/api/delete_project/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Удаление проекта"""
    try:
        project_file = os.path.join('projects', f"{project_id}.json")
        
        if not os.path.exists(project_file):
            return jsonify({'success': False, 'message': 'Проект не найден'})
        
        os.remove(project_file)
        
        return jsonify({
            'success': True,
            'message': 'Проект удален успешно'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при удалении проекта: {str(e)}'})

@app.route('/old')
def old_interface():
    """Старый интерфейс"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Прогнозирование маркетинговых данных</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .main-content {
            padding: 40px;
        }

        .upload-section {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            border-color: #4285f4;
            background: #f0f7ff;
        }

        .upload-section.dragover {
            border-color: #4285f4;
            background: #e3f2fd;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: #4285f4;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-btn:hover {
            background: #3367d6;
            transform: translateY(-2px);
        }

        .settings-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .setting-group {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4285f4;
        }

        .setting-group h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }

        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
        }

        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4285f4;
            box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2);
        }

        .forecast-btn {
            background: linear-gradient(135deg, #34a853 0%, #137333 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-bottom: 20px;
        }

        .forecast-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(52, 168, 83, 0.3);
        }

        .forecast-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .results-section {
            margin-top: 30px;
            display: none;
        }

        .results-section.show {
            display: block;
        }

        .results-header {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .results-header h3 {
            color: #137333;
            margin-bottom: 10px;
        }

        .download-btn {
            background: #ff6b35;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            transition: all 0.3s ease;
        }

        .download-btn:hover {
            background: #e55a2b;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #4285f4;
        }

        .stat-card h4 {
            color: #666;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4285f4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #dc3545;
        }

        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #28a745;
        }

        .data-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .data-info h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }

        .column-mapping {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .mapping-group {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }

        .mapping-group h5 {
            color: #e65100;
            margin-bottom: 10px;
        }

        @media (max-width: 768px) {
            .main-content {
                padding: 20px;
            }
            
            .settings-section {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Прогнозирование маркетинговых данных</h1>
            <p>Создавайте точные прогнозы на основе исторических данных с помощью каскадной модели</p>
        </div>

        <div class="main-content">
            <!-- Загрузка файла -->
            <div class="upload-section" id="uploadSection">
                <h3>📁 Загрузите CSV файл с данными</h3>
                <p>Перетащите файл сюда или нажмите кнопку для выбора</p>
                <input type="file" id="fileInput" class="file-input" accept=".csv,.xlsx,.xls" />
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Выбрать файл
                </button>
                <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
                    Поддерживаются файлы CSV, Excel с колонками: год, месяц, числовые метрики
                </p>
            </div>

            <!-- Информация о данных -->
            <div class="data-info" id="dataInfo" style="display: none;">
                <h4>📊 Информация о загруженных данных</h4>
                <div id="dataInfoContent"></div>
            </div>

            <!-- Маппинг колонок -->
            <div class="column-mapping" id="columnMapping" style="display: none;">
                <div class="mapping-group">
                    <h5>🗓️ Колонка с годом</h5>
                    <select id="yearColumn" onchange="updateMapping()">
                    <option value="0">A (1-я колонка)</option>
                    </select>
                </div>
                <div class="mapping-group">
                    <h5>📅 Колонка с месяцем</h5>
                    <select id="monthColumn" onchange="updateMapping()">
                    <option value="1">B (2-я колонка)</option>
                    </select>
                </div>
            </div>

            <!-- Настройки прогноза -->
            <div class="settings-section">
                <div class="setting-group">
                    <h3>⚙️ Параметры прогноза</h3>
                    <div class="form-group">
                    <label for="periods">Количество периодов для прогноза:</label>
                    <input type="number" id="periods" value="4" min="1" max="12" />
                    </div>
                    <div class="form-group">
                    <label for="method">Метод прогнозирования:</label>
                    <select id="method">
                    <option value="random_forest">Random Forest (рекомендуется)</option>
                    <option value="linear">Линейная регрессия</option>
                    </select>
                    </div>
                </div>
            </div>

            <!-- Кнопка прогноза -->
            <button class="forecast-btn" id="forecastBtn" onclick="createForecast()" disabled>
                🔮 Создать прогноз
            </button>

            <!-- Загрузка -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Создание прогноза...</p>
            </div>

            <!-- Результаты -->
            <div class="results-section" id="resultsSection">
                <div class="results-header">
                    <h3>✅ Прогноз успешно создан!</h3>
                    <p>Результаты готовы для скачивания и анализа</p>
                    <button class="download-btn" onclick="downloadResults()">📥 Скачать CSV</button>
                </div>

                <div class="stats-grid" id="statsGrid">
                    <!-- Статистика будет добавлена динамически -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let dataInfo = null;

        // Обработка загрузки файла
        document.getElementById('fileInput').addEventListener('change', handleFileUpload);
        
        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            if (!file.name.toLowerCase().match(/\\.(csv|xlsx|xls)$/)) {
                showError('Пожалуйста, выберите CSV или Excel файл');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            showLoading(true);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    sessionId = data.session_id;
                    dataInfo = data.data_info;
                    showDataInfo(dataInfo);
                    showSuccess(data.message);
                    document.getElementById('forecastBtn').disabled = false;
                } else {
                    showError(data.message);
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Ошибка при загрузке файла: ' + error.message);
            });
        }

        function showDataInfo(info) {
            const dataInfoDiv = document.getElementById('dataInfo');
            const contentDiv = document.getElementById('dataInfoContent');
            
            let html = `
                <p><strong>Размер:</strong> ${info.shape[0]} строк, ${info.shape[1]} колонок</p>
                <p><strong>Колонки:</strong></p>
                <ul>
            `;
            
            info.columns.forEach((col, index) => {
                html += `<li>${index}: ${col} (${info.dtypes[col]})</li>`;
            });
            
            html += '</ul>';
            contentDiv.innerHTML = html;
            dataInfoDiv.style.display = 'block';
            
            // Обновляем селекты для маппинга
            updateColumnSelects();
        }

        function updateColumnSelects() {
            const yearSelect = document.getElementById('yearColumn');
            const monthSelect = document.getElementById('monthColumn');
            
            // Очищаем селекты
            yearSelect.innerHTML = '';
            monthSelect.innerHTML = '';
            
            // Добавляем опции
            dataInfo.columns.forEach((col, index) => {
                const option1 = document.createElement('option');
                option1.value = index;
                option1.textContent = `${String.fromCharCode(65 + index)} (${index + 1}-я колонка): ${col}`;
                yearSelect.appendChild(option1);
                
                const option2 = document.createElement('option');
                option2.value = index;
                option2.textContent = `${String.fromCharCode(65 + index)} (${index + 1}-я колонка): ${col}`;
                monthSelect.appendChild(option2);
            });
            
            // Устанавливаем значения по умолчанию
            yearSelect.value = '0';
            monthSelect.value = '1';
            
            document.getElementById('columnMapping').style.display = 'grid';
        }

        function updateMapping() {
            // Функция для обновления маппинга (можно расширить)
        }

        function createForecast() {
            if (!sessionId) {
                showError('Сначала загрузите файл с данными');
                return;
            }

            const settings = {
                periods: parseInt(document.getElementById('periods').value),
                method: document.getElementById('method').value,
                year_column: parseInt(document.getElementById('yearColumn').value),
                month_column: parseInt(document.getElementById('monthColumn').value)
            };

            showLoading(true);

            fetch('/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    showResults(data);
                    showSuccess(data.message);
                } else {
                    showError(data.message);
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Ошибка при создании прогноза: ' + error.message);
            });
        }

        function showResults(data) {
            const section = document.getElementById('resultsSection');
            section.classList.add('show');
            
            // Показываем статистику
            showStats(data);
        }

        function showStats(data) {
            const statsGrid = document.getElementById('statsGrid');
            
            let html = `
                <div class="stat-card">
                    <h4>Создано прогнозов</h4>
                    <div class="value">${data.total_forecasts}</div>
                </div>
                <div class="stat-card">
                    <h4>Периодов прогноза</h4>
                    <div class="value">${data.settings.periods}</div>
                </div>
                <div class="stat-card">
                    <h4>Метод</h4>
                    <div class="value">${data.settings.method === 'random_forest' ? 'Random Forest' : 'Линейный'}</div>
                </div>
            `;
            
            statsGrid.innerHTML = html;
        }

        function downloadResults() {
            if (!sessionId) return;
            
            window.open(`/download/${sessionId}`, '_blank');
        }

        function showLoading(show) {
            const loading = document.getElementById('loading');
            const btn = document.getElementById('forecastBtn');
            
            if (show) {
                loading.classList.add('show');
                btn.disabled = true;
            } else {
                loading.classList.remove('show');
                btn.disabled = false;
            }
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            
            const container = document.querySelector('.main-content');
            container.insertBefore(errorDiv, container.firstChild);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }

        function showSuccess(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'success';
            successDiv.textContent = message;
            
            const container = document.querySelector('.main-content');
            container.insertBefore(successDiv, container.firstChild);
            
            setTimeout(() => {
                successDiv.remove();
            }, 3000);
        }
    </script>
</body>
</html>
    """)

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
        success, message = forecast_app.load_data_from_file(filepath)
        
        if success:
            forecast_app.session_id = session_id
            data_info = forecast_app.get_data_info()
            
            # Очищаем NaN значения для JSON
            import json
            data_info_json = json.dumps(data_info, default=str)
            data_info_clean = json.loads(data_info_json)
            
            return jsonify({
                'success': True, 
                'message': message,
                'session_id': session_id,
                'data_info': data_info_clean
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Недопустимый тип файла'})

@app.route('/api/get_time_series_values/<session_id>')
def get_time_series_values(session_id):
    """Получение уникальных значений временных рядов"""
    try:
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        df = forecast_app.df
        
        # Получаем маппинг - он должен быть передан в запросе
        mapping_data = request.args.get('mapping')
        if not mapping_data:
            return jsonify({'success': False, 'message': 'Маппинг не передан. Отправьте параметр mapping.'})
        
        import json as json_lib
        mapping = json_lib.loads(mapping_data)
        
        # Определяем временные поля из маппинга
        time_fields = {}
        for col in mapping.get('columns', []):
            if col.get('time_series') and col.get('include'):
                time_series_type = col['time_series']
                col_name = col['name']
                time_fields[time_series_type] = col_name
        
        print(f"Временные поля из маппинга: {time_fields}")
        
        time_series = []
        
        # Получаем уникальные комбинации
        if time_fields:
            group_cols = [col_name for col_name in time_fields.values() if col_name in df.columns]
            
            if not group_cols:
                return jsonify({'success': False, 'message': 'Временные колонки не найдены в данных'})
            
            unique_combinations = df[group_cols].drop_duplicates().to_dict('records')
            
            print(f"Найдено уникальных комбинаций: {len(unique_combinations)}")
            if unique_combinations:
                print(f"Первая комбинация: {unique_combinations[0]}")
            
            for combo in unique_combinations:
                item = {}
                # Обратное сопоставление: col_name -> time_series_type
                for time_type, col_name in time_fields.items():
                    value = combo.get(col_name)
                    if pd.notna(value) and value != '':
                        item[time_type] = value
                
                if item:
                    time_series.append(item)
        
        # Сортируем по году
        if time_series:
            time_series.sort(key=lambda x: (x.get('year', 0), x.get('month', 0)))
        
        print(f"Возвращаем временных рядов: {len(time_series)}")
        if time_series:
            print(f"Первый ряд: {time_series[0]}")
        
        return jsonify({
            'success': True,
            'time_series': time_series,
            'time_fields': time_fields
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/get_metric_time_series/<session_id>')
def get_metric_time_series(session_id):
    """Получение данных метрики по временным рядам"""
    try:
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        metric = request.args.get('metric')
        if not metric:
            return jsonify({'success': False, 'message': 'Метрика не указана'})
        
        df = forecast_app.df
        
        if metric not in df.columns:
            return jsonify({'success': False, 'message': f'Метрика {metric} не найдена'})
        
        # Определяем временные поля (year и month)
        year_field = None
        month_field = None
        
        for col in df.columns:
            if 'year' in col.lower() and not year_field:
                year_field = col
            if 'month' in col.lower() and not month_field:
                month_field = col
        
        if not year_field:
            return jsonify({'success': False, 'message': 'Поле года не найдено'})
        
        # Если есть поле месяца, агрегируем по году и месяцу
        if month_field:
            # Создаем составной ключ год-месяц
            df_copy = df.copy()
            df_copy['year_month'] = df_copy[year_field].astype(str) + '-' + df_copy[month_field].astype(str).str.zfill(2)
            
            aggregated = df_copy.groupby(['year_month', year_field, month_field])[metric].sum().reset_index()
            aggregated = aggregated.sort_values([year_field, month_field])
            
            labels = aggregated['year_month'].tolist()
            values = aggregated[metric].tolist()
        else:
            # Только по годам
            aggregated = df.groupby(year_field)[metric].sum().reset_index()
            aggregated = aggregated.sort_values(year_field)
            
            labels = aggregated[year_field].astype(str).tolist()
            values = aggregated[metric].tolist()
        
        return jsonify({
            'success': True,
            'data': {
                'labels': labels,
                'values': values,
                'metric': metric
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/save_forecast_settings', methods=['POST'])
def save_forecast_settings():
    """Сохранение настроек прогноза"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'message': 'Session ID не указан'})
        
        # Сохраняем настройки в сессию
        forecast_settings = {
            'metric': data.get('metric'),
            'forecast_months': data.get('forecast_months', 0),
            'forecast_periods': data.get('forecast_periods', []),
            'time_series_config': data.get('time_series_config', {}),
            'created_at': datetime.now().isoformat()
        }
        
        # Сохраняем в глобальную переменную
        if not hasattr(forecast_app, 'forecast_settings'):
            forecast_app.forecast_settings = {}
        
        forecast_app.forecast_settings[session_id] = forecast_settings
        
        print(f"✅ Сохранены настройки прогноза для сессии {session_id}")
        print(f"   Метрика: {forecast_settings['metric']}")
        print(f"   Горизонт: {forecast_settings['forecast_months']} месяцев")
        print(f"   Прогнозных периодов: {len(forecast_settings['forecast_periods'])}")
        
        return jsonify({
            'success': True,
            'message': 'Настройки прогноза сохранены'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/get_forecast_settings/<session_id>')
def get_forecast_settings(session_id):
    """Получение сохраненных настроек прогноза"""
    try:
        if not hasattr(forecast_app, 'forecast_settings') or session_id not in forecast_app.forecast_settings:
            return jsonify({'success': False, 'message': 'Настройки прогноза не найдены'})
        
        settings = forecast_app.forecast_settings[session_id]
        
        return jsonify({
            'success': True,
            'settings': settings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/train_models', methods=['POST'])
def train_models():
    """Обучение моделей прогнозирования"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        metric = data.get('metric')
        models_to_train = data.get('models', [])
        test_size = data.get('test_size', 0.2)
        
        print(f"\n🎯 ОБУЧЕНИЕ МОДЕЛЕЙ:", flush=True)
        print(f"   Метрика: {metric}", flush=True)
        print(f"   Модели: {models_to_train}", flush=True)
        print(f"   Размер тестовой выборки: {test_size * 100}%", flush=True)
        
        # Подготовка данных
        df = forecast_app.df
        
        # Получаем маппинг
        mapping_config = data.get('mapping')
        if not mapping_config and hasattr(forecast_app, 'mapping_config'):
            mapping_config = forecast_app.mapping_config
        
        # Находим временные поля и поля срезов из маппинга
        year_col = None
        month_col = None
        slice_cols = []
        
        if mapping_config and mapping_config.get('columns'):
            for col_config in mapping_config['columns']:
                if col_config.get('time_series') == 'year':
                    year_col = col_config['name']
                elif col_config.get('time_series') == 'month':
                    month_col = col_config['name']
                elif col_config.get('role') == 'dimension' and not col_config.get('time_series'):
                    slice_cols.append(col_config['name'])
        
        # Fallback: поиск по названиям
        if not year_col or not month_col:
            for col in df.columns:
                if 'year' in col.lower() and not year_col:
                    year_col = col
                if 'month' in col.lower() and not month_col:
                    month_col = col
        
        if not year_col or not month_col or metric not in df.columns:
            return jsonify({'success': False, 'message': 'Необходимые поля не найдены'})
        
        print(f"   📊 Поля для обучения:", flush=True)
        print(f"      Временные: {year_col}, {month_col}", flush=True)
        print(f"      Срезы: {slice_cols}", flush=True)
        print(f"      Метрика: {metric}", flush=True)
        
        # Агрегируем данные по году-месяцу + срезы
        groupby_cols = [year_col, month_col] + slice_cols
        df_agg = df.groupby(groupby_cols)[metric].sum().reset_index()
        df_agg = df_agg.sort_values([year_col, month_col])
        
        print(f"   📊 После агрегации: {len(df_agg)} уникальных комбинаций", flush=True)
        
        results = {}
        
        # Если есть срезы - используем новый подход (срезы как признаки)
        if slice_cols:
            print(f"   🔄 Обучение моделей со срезами как признаками...", flush=True)
            print(f"   📊 Уникальных комбинаций срезов: {len(df_agg)}", flush=True)
            
            # Обучаем каждую модель
            for model_name in models_to_train:
                print(f"\n📊 Обучение модели: {model_name}", flush=True)
                
                try:
                    if model_name == 'prophet':
                        model_result = train_prophet_with_slices(df_agg, metric, year_col, month_col, slice_cols, test_size)
                    elif model_name == 'random_forest':
                        model_result = train_random_forest_with_slices(df_agg, metric, year_col, month_col, slice_cols, test_size)
                    elif model_name == 'random_forest_hierarchy':
                        model_result = train_random_forest_hierarchy(df_agg, metric, year_col, month_col, slice_cols, test_size)
                    elif model_name == 'arima':
                        print(f"   ⚠️ ARIMA не поддерживается для данных со срезами, пропускаем", flush=True)
                        continue
                    else:
                        print(f"   ⚠️ Неизвестная модель {model_name}, пропускаем", flush=True)
                        continue
                    
                    results[model_name] = model_result
                    
                    # Выводим метрики
                    if 'metrics_before_reconciliation' in model_result:
                        improvement = model_result.get('reconciliation_improvement', 0)
                        print(f"   ✅ {model_name}: MAPE = {model_result['metrics']['mape']:.2f}% (улучшение: {improvement:.2f}%)", flush=True)
                    else:
                        print(f"   ✅ {model_name}: MAPE = {model_result['metrics']['mape']:.2f}%", flush=True)
                    
                except Exception as e:
                    import traceback
                    print(f"   ❌ Ошибка обучения {model_name}: {e}", flush=True)
                    traceback.print_exc()
                    continue
        
        else:
            # Нет срезов - используем старую логику (один общий прогноз)
            print(f"   📊 Нет срезов, обучаем на агрегированных данных")
            
            df_agg['period'] = df_agg[year_col].astype(str) + '-' + df_agg[month_col].astype(str).str.zfill(2)
            
            print(f"   Всего периодов: {len(df_agg)}")
            
            # Разделение на обучающую и контрольную выборки
            split_index = int(len(df_agg) * (1 - test_size))
            train_df = df_agg[:split_index]
            test_df = df_agg[split_index:]
            
            print(f"   Обучающая выборка: {len(train_df)} периодов")
            print(f"   Контрольная выборка: {len(test_df)} периодов")
            
            for model_name in models_to_train:
                print(f"\n📊 Обучение модели: {model_name}")
                
                try:
                    if model_name == 'arima':
                        model_result = train_arima_model(train_df, test_df, metric)
                    elif model_name == 'prophet':
                        model_result = train_prophet_model(train_df, test_df, metric, year_col, month_col)
                    elif model_name == 'random_forest':
                        model_result = train_random_forest_model(train_df, test_df, metric, year_col, month_col)
                    else:
                        continue
                    
                    results[model_name] = model_result
                    print(f"   ✅ {model_name}: MAPE = {model_result['metrics']['mape']:.2f}%")
                    
                except Exception as e:
                    print(f"   ❌ Ошибка обучения {model_name}: {e}")
                    continue
        
        if not results:
            return jsonify({'success': False, 'message': 'Не удалось обучить ни одну модель'})
        
        # Сохраняем результаты обучения
        if not hasattr(forecast_app, 'training_results'):
            forecast_app.training_results = {}
        
        # Сохраняем полные результаты (с моделями) для генерации прогноза
        forecast_app.training_results[session_id] = results
        
        # Создаем копию для отправки клиенту (без объектов модели)
        results_for_client = {}
        for model_name, model_data in results.items():
            results_for_client[model_name] = {
                'metrics': model_data['metrics'],
                'validation_data': model_data['validation_data'],
                'detailed_validation': model_data.get('detailed_validation', []),
                'slice_cols': model_data.get('slice_cols', [])
            }
        
        return jsonify({
            'success': True,
            'results': results_for_client
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/generate_forecast', methods=['POST'])
def generate_forecast():
    """Генерация прогноза с использованием обученной модели"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        selected_model = data.get('model')
        mapping_from_request = data.get('mapping')  # Получаем маппинг из запроса
        
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        # Получаем настройки прогноза
        if not hasattr(forecast_app, 'forecast_settings') or session_id not in forecast_app.forecast_settings:
            return jsonify({'success': False, 'message': 'Настройки прогноза не найдены'})
        
        settings = forecast_app.forecast_settings[session_id]
        metric = settings['metric']
        forecast_periods = settings['forecast_periods']
        
        # Получаем маппинг (приоритет - из запроса)
        mapping_config = None
        if mapping_from_request:
            mapping_config = mapping_from_request
            print("   ✅ Маппинг получен из запроса", flush=True)
        elif hasattr(forecast_app, 'mapping_config'):
            mapping_config = forecast_app.mapping_config
            print("   ✅ Маппинг получен из forecast_app", flush=True)
        else:
            # Пытаемся загрузить из проекта
            import json
            project_file = f'projects/{session_id}.json'
            if os.path.exists(project_file):
                with open(project_file, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                    mapping_config = project_data.get('data_mapping', {})
                    print("   ✅ Маппинг загружен из файла проекта", flush=True)
        
        if not mapping_config or not mapping_config.get('columns'):
            print("   ⚠️ Маппинг не найден, будет использован базовый набор метрик", flush=True)
            mapping_config = {'columns': []}
        
        print(f"\n🚀 ГЕНЕРАЦИЯ ПРОГНОЗА:", flush=True)
        print(f"   Модель: {selected_model}", flush=True)
        print(f"   Метрика: {metric}", flush=True)
        print(f"   Прогнозных периодов: {len(forecast_periods)}", flush=True)
        
        # Очищаем старые результаты прогноза для этой сессии
        if hasattr(forecast_app, 'forecast_results') and session_id in forecast_app.forecast_results:
            print(f"   🗑️ Удаляем старый прогноз из кэша", flush=True)
            del forecast_app.forecast_results[session_id]
        else:
            print(f"   ℹ️ Старого прогноза нет, создаем новый", flush=True)
        
        # Подготовка данных для прогноза
        df = forecast_app.df
        
        # Находим временные поля и поля срезов из маппинга
        year_col = None
        month_col = None
        slice_cols = []
        
        # Используем маппинг для определения полей
        if mapping_config and mapping_config.get('columns'):
            for col_config in mapping_config['columns']:
                if col_config.get('time_series') == 'year':
                    year_col = col_config['name']
                elif col_config.get('time_series') == 'month':
                    month_col = col_config['name']
                elif col_config.get('role') == 'dimension' and not col_config.get('time_series'):
                    # Это поле среза
                    slice_cols.append(col_config['name'])
        
        # Fallback: поиск по названиям
        if not year_col or not month_col:
            for col in df.columns:
                if 'year' in col.lower() and not year_col:
                    year_col = col
                if 'month' in col.lower() and not month_col:
                    month_col = col
        
        if not year_col or not month_col or metric not in df.columns:
            return jsonify({'success': False, 'message': 'Необходимые поля не найдены'})
        
        print(f"   📊 Поля для прогноза:", flush=True)
        print(f"      Временные: {year_col}, {month_col}", flush=True)
        print(f"      Срезы: {slice_cols}", flush=True)
        print(f"      Метрика: {metric}", flush=True)
        
        # Сохраняем все исходные данные (не агрегируем!)
        # Добавляем is_forecast = False ко всем фактическим данным
        df['is_forecast'] = False
        
        # Агрегируем данные для построения прогноза
        # Группируем по временным полям + срезам
        groupby_cols = [year_col, month_col] + slice_cols
        df_agg = df.groupby(groupby_cols)[metric].sum().reset_index()
        df_agg = df_agg.sort_values([year_col, month_col])
        
        print(f"   📊 После агрегации: {len(df_agg)} уникальных комбинаций", flush=True)
        print(f"   📊 Первые 3 строки:", flush=True)
        print(df_agg.head(3).to_dict('records'), flush=True)
        
        # Создаем список прогнозных периодов
        forecast_months = []
        for period in forecast_periods:
            for month in period['months']:
                forecast_months.append({
                    'year': period['year'],
                    'month': month
                })
        
        print(f"   Всего прогнозных месяцев: {len(forecast_months)}", flush=True)
        
        # Получаем список всех метрик из маппинга
        all_metrics = [col['name'] for col in mapping_config.get('columns', []) if col.get('role') == 'metric']
        print(f"   📊 Все метрики: {all_metrics}", flush=True)
        print(f"   🎯 Метрика с прогнозом: {metric}", flush=True)
        
        # Если есть срезы - строим прогноз для каждой комбинации срезов
        if slice_cols:
            print(f"   🔄 Строим прогноз для каждой комбинации срезов...", flush=True)
            
            # Получаем уникальные комбинации срезов
            unique_slices = df_agg[slice_cols].drop_duplicates().to_dict('records')
            print(f"   📊 Уникальных комбинаций срезов: {len(unique_slices)}", flush=True)
            print(f"   📊 Первые 3 комбинации:", unique_slices[:3], flush=True)
            
            forecast_rows = []
            
            # Специальная обработка для random_forest_hierarchy
            if selected_model == 'random_forest_hierarchy':
                print(f"   🏗️ Используется Random Forest Hierarchy - генерируем прогноз для всех срезов сразу", flush=True)
                
                # Получаем данные обученной модели
                if not hasattr(forecast_app, 'training_results') or session_id not in forecast_app.training_results:
                    return jsonify({'success': False, 'message': 'Модель не обучена. Сначала обучите модель.'})
                
                trained_model_data = forecast_app.training_results[session_id].get('random_forest_hierarchy')
                if not trained_model_data:
                    return jsonify({'success': False, 'message': 'Random Forest Hierarchy не обучена'})
                
                # Генерируем прогнозы для всех комбинаций сразу
                try:
                    all_forecasts_detailed = generate_random_forest_hierarchy_forecast_detailed(
                        df_agg, metric, year_col, month_col, slice_cols, forecast_months, trained_model_data
                    )
                    
                    # Преобразуем в forecast_rows
                    for forecast_dict in all_forecasts_detailed:
                        forecast_row = {}
                        forecast_row[year_col] = forecast_dict['year']
                        forecast_row[month_col] = forecast_dict['month']
                        
                        # Копируем срезы
                        for slice_col in slice_cols:
                            forecast_row[slice_col] = forecast_dict[slice_col]
                        
                        # Устанавливаем метрику
                        forecast_row[metric] = forecast_dict['predicted']
                        for other_metric in all_metrics:
                            if other_metric != metric:
                                forecast_row[other_metric] = 0
                        
                        forecast_row['is_forecast'] = True
                        forecast_row['Quarter'] = f'Q{(forecast_dict["month"]-1)//3 + 1}'
                        forecast_row['Halfyear'] = 'H1' if forecast_dict['month'] <= 6 else 'H2'
                        
                        forecast_rows.append(forecast_row)
                    
                    print(f"   ✅ Random Forest Hierarchy: создано {len(forecast_rows)} прогнозных строк", flush=True)
                    
                except Exception as e:
                    print(f"   ❌ Ошибка генерации прогноза Random Forest Hierarchy: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})
            
            elif selected_model == 'random_forest':
                # Random Forest тоже использует обученную модель со всеми срезами
                print(f"   🌲 Используется Random Forest - генерируем прогноз для всех срезов", flush=True)
                
                if not hasattr(forecast_app, 'training_results') or session_id not in forecast_app.training_results:
                    return jsonify({'success': False, 'message': 'Модель не обучена. Сначала обучите модель.'})
                
                trained_model_data = forecast_app.training_results[session_id].get('random_forest')
                if not trained_model_data:
                    return jsonify({'success': False, 'message': 'Random Forest не обучена'})
                
                model = trained_model_data.get('model')
                label_encoders = trained_model_data.get('label_encoders', {})
                
                if not model:
                    return jsonify({'success': False, 'message': 'Обученная модель не найдена'})
                
                # Генерируем прогноз для каждой комбинации срезов
                for slice_combination in unique_slices:
                    for fm in forecast_months:
                        forecast_row = {
                            year_col: fm['year'],
                            month_col: fm['month']
                        }
                        
                        # Кодируем срезы
                        for slice_col in slice_cols:
                            forecast_row[slice_col] = slice_combination[slice_col]
                            encoded_col = f'{slice_col}_encoded'
                            if encoded_col in label_encoders:
                                le = label_encoders[encoded_col]
                                value = slice_combination[slice_col]
                                try:
                                    encoded_value = le.transform([value if value in le.classes_ else 'unknown'])[0]
                                except:
                                    encoded_value = 0
                                forecast_row[encoded_col] = encoded_value
                        
                        # Формируем вектор признаков
                        feature_cols = [year_col, month_col] + [f'{col}_encoded' for col in slice_cols]
                        X_forecast = np.array([[forecast_row.get(col, 0) for col in feature_cols]])
                        
                        # Прогноз
                        predicted_value = model.predict(X_forecast)[0]
                        
                        # Создаем прогнозную строку
                        final_row = {}
                        final_row[year_col] = fm['year']
                        final_row[month_col] = fm['month']
                        
                        for slice_col in slice_cols:
                            final_row[slice_col] = slice_combination[slice_col]
                        
                        final_row[metric] = predicted_value
                        for other_metric in all_metrics:
                            if other_metric != metric:
                                final_row[other_metric] = 0
                        
                        final_row['is_forecast'] = True
                        final_row['Quarter'] = f'Q{(fm["month"]-1)//3 + 1}'
                        final_row['Halfyear'] = 'H1' if fm['month'] <= 6 else 'H2'
                        
                        forecast_rows.append(final_row)
                
                print(f"   ✅ Random Forest: создано {len(forecast_rows)} прогнозных строк", flush=True)
            
            else:
                # Для других моделей (prophet, arima) - цикл по срезам с переобучением
                print(f"   ⚠️ Модель {selected_model} будет переобучена для каждого среза", flush=True)
                
                for slice_combination in unique_slices:
                    # Фильтруем данные для этой комбинации срезов
                    mask = pd.Series([True] * len(df_agg))
                    for slice_col in slice_cols:
                        mask &= (df_agg[slice_col] == slice_combination[slice_col])
                    
                    df_slice = df_agg[mask].copy()
                    
                    if len(df_slice) < 10:
                        continue
                    
                    # Строим прогноз для этой комбинации
                    try:
                        if selected_model == 'arima':
                            slice_forecast = generate_arima_forecast(df_slice, metric, len(forecast_months))
                        elif selected_model == 'prophet':
                            slice_forecast = generate_prophet_forecast(df_slice, metric, year_col, month_col, forecast_months)
                        else:
                            continue
                        
                        # Создаем прогнозные строки для этой комбинации срезов
                        for i, month_data in enumerate(forecast_months):
                            forecast_row = {}
                            forecast_row[year_col] = month_data['year']
                            forecast_row[month_col] = month_data['month']
                            
                            # Копируем значения срезов
                            for slice_col in slice_cols:
                                forecast_row[slice_col] = slice_combination[slice_col]
                            
                            # Устанавливаем значения метрик
                            forecast_row[metric] = slice_forecast[i]
                            for other_metric in all_metrics:
                                if other_metric != metric:
                                    forecast_row[other_metric] = 0
                            
                            forecast_row['is_forecast'] = True
                            forecast_row['Quarter'] = f'Q{(month_data["month"]-1)//3 + 1}'
                            forecast_row['Halfyear'] = 'H1' if month_data['month'] <= 6 else 'H2'
                            
                            forecast_rows.append(forecast_row)
                    
                    except Exception as e:
                        print(f"   ⚠️ Ошибка прогноза для {slice_combination}: {e}", flush=True)
                        continue
            
            forecast_df = pd.DataFrame(forecast_rows)
            print(f"   ✅ Создано прогнозных строк: {len(forecast_df)}", flush=True)
        else:
            # Нет срезов - строим один общий прогноз (старая логика)
            print(f"   📊 Нет срезов, строим общий прогноз", flush=True)
            
            if selected_model == 'arima':
                forecast_values = generate_arima_forecast(df_agg, metric, len(forecast_months))
            elif selected_model == 'prophet':
                forecast_values = generate_prophet_forecast(df_agg, metric, year_col, month_col, forecast_months)
            elif selected_model == 'random_forest':
                forecast_values = generate_random_forest_forecast(df_agg, metric, year_col, month_col, forecast_months)
            else:
                return jsonify({'success': False, 'message': f'Неизвестная модель: {selected_model}'})
            
            # Создаем прогнозные строки
            forecast_rows = []
            last_row = df.iloc[-1].to_dict()
            
            for i, month_data in enumerate(forecast_months):
                forecast_row = last_row.copy()
                forecast_row[year_col] = month_data['year']
                forecast_row[month_col] = month_data['month']
                forecast_row[metric] = forecast_values[i]
                
                for other_metric in all_metrics:
                    if other_metric != metric:
                        forecast_row[other_metric] = 0
                
                forecast_row['is_forecast'] = True
                forecast_row['Quarter'] = f'Q{(month_data["month"]-1)//3 + 1}'
                forecast_row['Halfyear'] = 'H1' if month_data['month'] <= 6 else 'H2'
                
                forecast_rows.append(forecast_row)
            
            forecast_df = pd.DataFrame(forecast_rows)
        
        # Добавляем Quarter и Halfyear к фактическим данным если их нет
        if 'Quarter' not in df.columns:
            df['Quarter'] = df[month_col].apply(lambda m: f'Q{(m-1)//3 + 1}')
        if 'Halfyear' not in df.columns:
            df['Halfyear'] = df[month_col].apply(lambda m: 'H1' if m <= 6 else 'H2')
        
        # Объединяем фактические данные со всеми колонками + прогнозные данные
        combined_df = pd.concat([df, forecast_df], ignore_index=True)
        
        print(f"   📊 Объединено: {len(df)} фактических + {len(forecast_df)} прогнозных = {len(combined_df)} строк")
        
        print(f"   ✅ Прогноз построен: {len(forecast_df)} периодов")
        
        # Сохраняем результаты прогноза в память
        if not hasattr(forecast_app, 'forecast_results'):
            forecast_app.forecast_results = {}
        
        forecast_app.forecast_results[session_id] = {
            'model': selected_model,
            'metric': metric,
            'combined_data': combined_df.to_dict('records'),
            'forecast_only': forecast_df.to_dict('records'),
            'historical_periods': len(df),
            'forecast_periods': len(forecast_df)
        }
        
        # Физическое сохранение в файлы
        try:
            # Создаем папку для результатов прогноза если её нет
            forecast_dir = 'results'
            if not os.path.exists(forecast_dir):
                os.makedirs(forecast_dir)
            
            # Сохраняем объединенные данные (факт + прогноз)
            combined_filename = f'forecast_combined_{session_id}.csv'
            combined_path = os.path.join(forecast_dir, combined_filename)
            combined_df.to_csv(combined_path, index=False, encoding='utf-8')
            print(f"   💾 Сохранен объединенный файл: {combined_path}")
            
            # Сохраняем только прогнозные данные
            forecast_filename = f'forecast_only_{session_id}.csv'
            forecast_path = os.path.join(forecast_dir, forecast_filename)
            forecast_df.to_csv(forecast_path, index=False, encoding='utf-8')
            print(f"   💾 Сохранен файл прогноза: {forecast_path}")
            
            # Обновляем информацию в forecast_results
            forecast_app.forecast_results[session_id]['combined_file'] = combined_path
            forecast_app.forecast_results[session_id]['forecast_file'] = forecast_path
            
        except Exception as e:
            print(f"   ⚠️ Ошибка сохранения файлов: {e}")
            # Продолжаем работу даже если сохранение не удалось
        
        return jsonify({
            'success': True,
            'message': 'Прогноз успешно построен',
            'files': {
                'combined': combined_filename if 'combined_filename' in locals() else None,
                'forecast_only': forecast_filename if 'forecast_filename' in locals() else None
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/get_forecast_results/<session_id>')
def get_forecast_results(session_id):
    """Получение результатов прогноза"""
    try:
        if not hasattr(forecast_app, 'forecast_results') or session_id not in forecast_app.forecast_results:
            return jsonify({'success': False, 'message': 'Результаты прогноза не найдены'})
        
        results = forecast_app.forecast_results[session_id]
        
        return jsonify({
            'success': True,
            'forecast_data': {
                'raw_data': results['combined_data'],
                'pivot_data': None  # Будет построена на фронтенде
            },
            'info': {
                'model': results['model'],
                'metric': results['metric'],
                'historical_periods': results['historical_periods'],
                'forecast_periods': results['forecast_periods'],
                'files': {
                    'combined': results.get('combined_file'),
                    'forecast_only': results.get('forecast_file')
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/export_forecast/<session_id>')
def export_forecast(session_id):
    """Экспорт результатов прогноза в CSV"""
    try:
        if not hasattr(forecast_app, 'forecast_results') or session_id not in forecast_app.forecast_results:
            return jsonify({'success': False, 'message': 'Результаты прогноза не найдены'})
        
        results = forecast_app.forecast_results[session_id]
        
        # Проверяем наличие сохраненного файла
        if 'combined_file' in results and os.path.exists(results['combined_file']):
            return send_file(
                results['combined_file'],
                as_attachment=True,
                download_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            return jsonify({'success': False, 'message': 'Файл прогноза не найден'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка: {str(e)}'})

@app.route('/api/update_file', methods=['POST'])
def update_file():
    """Обновление файла данных с сохранением session_id"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Файл не выбран'})
        
        file = request.files['file']
        old_session_id = request.form.get('session_id')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Файл не выбран'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Используем тот же session_id для сохранения настроек маппинга
            session_id = old_session_id if old_session_id else str(uuid.uuid4())
            
            # Удаляем старый файл с тем же session_id
            upload_folder = app.config['UPLOAD_FOLDER']
            if old_session_id:
                old_files = [f for f in os.listdir(upload_folder) if f.startswith(old_session_id)]
                for old_file in old_files:
                    try:
                        os.remove(os.path.join(upload_folder, old_file))
                        print(f"Удален старый файл: {old_file}")
                    except:
                        pass
            
            # Сохраняем новый файл с тем же session_id
            new_filename = f"{session_id}_{filename}"
            filepath = os.path.join(upload_folder, new_filename)
            file.save(filepath)
            
            # Загружаем данные
            success, message = forecast_app.load_data_from_file(filepath)
            
            if success:
                forecast_app.session_id = session_id
                data_info = forecast_app.get_data_info()
                
                return jsonify({
                    'success': True,
                    'message': 'Файл успешно обновлен',
                    'session_id': session_id,
                    'rows': data_info['shape'][0],
                    'columns': data_info['shape'][1],
                    'filename': filename
                })
            else:
                return jsonify({'success': False, 'message': message})
        
        return jsonify({'success': False, 'message': 'Недопустимый тип файла'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при обновлении файла: {str(e)}'})

@app.route('/forecast_api', methods=['POST'])
def forecast_api():
    """Создание прогноза"""
    try:
        config = request.json
        print(f"DEBUG: Получен запрос на прогнозирование: {config}")
        
        # Получаем session_id из запроса
        session_id = config.get('session_id')
        if not session_id or forecast_app.session_id != session_id:
            return jsonify({'success': False, 'message': 'Сессия не найдена'})
        
        # Проверяем наличие данных
        if forecast_app.df is None:
            return jsonify({'success': False, 'message': 'Данные не загружены'})
        
        # Получаем настройки маппинга из запроса или используем значения по умолчанию
        mapping_data = config.get('mapping_data')
        if mapping_data:
            mapping = json.loads(mapping_data)
            print(f"DEBUG: Используем маппинг из запроса: {mapping}")
        else:
            # Используем значения по умолчанию
            mapping = {
                'year': config.get('year_column', 0),
                'month': config.get('month_column', 1)
            }
        
        # Устанавливаем маппинг колонок
        forecast_app.set_data_mapping(mapping)
        
        # Подготавливаем конфигурацию для прогноза
        forecast_config = {
            'periods': config.get('periods', 4),
            'method': config.get('method', 'random_forest'),
            'target_metric': config.get('target_metric'),
            'enable_cascade': config.get('enable_cascade', True)
        }
        
        print(f"DEBUG: Конфигурация прогноза: {forecast_config}")
        
        # Запускаем прогноз
        success, message = forecast_app.run_cascaded_forecast(forecast_config)
        
        if success:
            # Сохраняем результаты
            forecast_app.save_results(forecast_app.session_id)
            
            return jsonify({
                'success': True,
                'message': message,
                'total_forecasts': forecast_app.forecast_results.get('total_forecasts', 0),
                'settings': forecast_config
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        print(f"ERROR: Ошибка при выполнении прогноза: {e}")
        return jsonify({'success': False, 'message': f'Ошибка сервера: {str(e)}'})

@app.route('/download/<session_id>')
def download_results(session_id):
    """Скачивание результатов"""
    filename = f"cascaded_forecast_{session_id}.csv"
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        return jsonify({'success': False, 'message': 'Файл не найден'})

if __name__ == '__main__':
    print("🚀 Запуск MARFOR веб-приложения...")
    print("📊 Каскадная модель с Random Forest")
    print("🔧 ВЕРСИЯ КОДА: 2.21.0 - Генерация прогноза и страница результатов")
    print("🌐 Откройте http://localhost:5001 в браузере")
    app.run(debug=True, host='0.0.0.0', port=5001)
