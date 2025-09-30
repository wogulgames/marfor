#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой интерфейс для универсальной программы прогнозирования
"""

import os
import sys
from universal_forecast_program import UniversalForecaster

def get_user_input():
    """Получение параметров от пользователя"""
    print("Универсальная программа прогнозирования")
    print("="*50)
    
    # Путь к файлу
    csv_file = input("Введите путь к CSV файлу (или нажмите Enter для 'Marketing Budjet Emulation - raw2.csv'): ").strip()
    if not csv_file:
        csv_file = 'Marketing Budjet Emulation - raw2.csv'
    
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return None
    
    # Количество периодов для прогноза
    try:
        periods = int(input("Количество периодов для прогноза (по умолчанию 4): ") or "4")
    except ValueError:
        periods = 4
    
    # Название выходного файла
    output_file = input("Название выходного файла (или нажмите Enter для 'Universal_Forecast_Results.csv'): ").strip()
    if not output_file:
        output_file = 'Universal_Forecast_Results.csv'
    
    return {
        'csv_file': csv_file,
        'periods': periods,
        'output_file': output_file
    }

def run_forecast(params):
    """Запуск прогнозирования"""
    try:
        # Инициализация
        forecaster = UniversalForecaster(params['csv_file'])
        
        # Загрузка данных
        print(f"\nЗагрузка данных из {params['csv_file']}...")
        forecaster.load_data()
        
        # Автоматическое определение колонок
        print("\nАвтоматическое определение колонок...")
        forecaster.auto_detect_columns()
        
        # Показываем найденные колонки
        print(f"Временные колонки: {forecaster.date_columns}")
        print(f"Колонки для прогнозирования: {forecaster.target_columns}")
        
        # Подтверждение от пользователя
        confirm = input("\nПродолжить с этими колонками? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', 'да', 'д']:
            print("Прогнозирование отменено")
            return
        
        # Обучение моделей с валидацией
        print("\nОбучение моделей с валидацией...")
        forecaster.train_models_with_validation()
        
        # Создание прогноза
        print(f"\nСоздание прогноза на {params['periods']} периодов...")
        forecaster.create_forecast(forecast_periods=params['periods'])
        
        # Сохранение результатов
        print(f"\nСохранение результатов в {params['output_file']}...")
        forecaster.save_forecast(params['output_file'])
        
        # Генерация отчета
        forecaster.generate_model_report()
        
        # Визуализация
        try:
            forecaster.plot_forecast()
        except Exception as e:
            print(f"Ошибка при создании графиков: {e}")
        
        print(f"\nПрогнозирование завершено успешно!")
        print(f"Результаты сохранены в файл: {params['output_file']}")
        
    except Exception as e:
        print(f"Ошибка при выполнении прогнозирования: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Основная функция"""
    try:
        # Получение параметров
        params = get_user_input()
        if params is None:
            return
        
        # Запуск прогнозирования
        run_forecast(params)
        
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

if __name__ == "__main__":
    main()
