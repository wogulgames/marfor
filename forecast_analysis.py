#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа для анализа качества прогноза и визуализации результатов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class ForecastAnalyzer:
    def __init__(self, csv_file):
        """
        Инициализация анализатора прогноза
        
        Args:
            csv_file (str): Путь к CSV файлу с прогнозом
        """
        self.csv_file = csv_file
        self.df = None
        
    def load_data(self):
        """Загрузка данных"""
        print("Загрузка данных для анализа...")
        self.df = pd.read_csv(self.csv_file)
        
        # Создание временного индекса
        self.df['date'] = pd.to_datetime(self.df[['year', 'month']].assign(day=1))
        
        print(f"Загружено {len(self.df)} записей")
        print(f"Период данных: {self.df['date'].min()} - {self.df['date'].max()}")
        
    def analyze_trends(self):
        """Анализ трендов по основным метрикам"""
        print("\n" + "="*60)
        print("АНАЛИЗ ТРЕНДОВ")
        print("="*60)
        
        # Агрегируем данные по месяцам
        monthly_data = self.df.groupby(['year', 'month']).agg({
            'revenue_total': 'sum',
            'traffic_total': 'sum',
            'transacitons_total': 'sum',
            'ads_cost': 'sum',
            'mar_cost': 'sum'
        }).reset_index()
        
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        
        # Разделяем на исторические данные и прогноз
        historical = monthly_data[monthly_data['date'] < '2026-09-01']
        forecast = monthly_data[monthly_data['date'] >= '2026-09-01']
        
        print("Исторические данные (2023-2026.08):")
        print(f"  Средняя месячная выручка: {historical['revenue_total'].mean():,.0f}")
        print(f"  Средний месячный трафик: {historical['traffic_total'].mean():,.0f}")
        print(f"  Средние месячные транзакции: {historical['transacitons_total'].mean():,.0f}")
        
        print("\nПрогноз (2026.09-2026.12):")
        print(f"  Средняя месячная выручка: {forecast['revenue_total'].mean():,.0f}")
        print(f"  Средний месячный трафик: {forecast['traffic_total'].mean():,.0f}")
        print(f"  Средние месячные транзакции: {forecast['transacitons_total'].mean():,.0f}")
        
        # Расчет роста
        revenue_growth = ((forecast['revenue_total'].mean() - historical['revenue_total'].mean()) / 
                         historical['revenue_total'].mean()) * 100
        traffic_growth = ((forecast['traffic_total'].mean() - historical['traffic_total'].mean()) / 
                         historical['traffic_total'].mean()) * 100
        
        print(f"\nПрогнозируемый рост:")
        print(f"  Выручка: {revenue_growth:+.1f}%")
        print(f"  Трафик: {traffic_growth:+.1f}%")
        
        return monthly_data, historical, forecast
    
    def analyze_categories(self):
        """Анализ по категориям"""
        print("\n" + "="*60)
        print("АНАЛИЗ ПО КАТЕГОРИЯМ")
        print("="*60)
        
        # Анализ прогноза по категориям
        forecast_data = self.df[(self.df['year'] == 2026) & (self.df['month'] >= 9)]
        
        category_analysis = forecast_data.groupby('category').agg({
            'revenue_total': 'sum',
            'traffic_total': 'sum',
            'transacitons_total': 'sum'
        }).sort_values('revenue_total', ascending=False)
        
        print("Топ-15 категорий по прогнозируемой выручке:")
        for i, (category, row) in enumerate(category_analysis.head(15).iterrows(), 1):
            print(f"{i:2d}. {category}: {row['revenue_total']:,.0f}")
        
        return category_analysis
    
    def analyze_subdivisions(self):
        """Анализ по подразделениям"""
        print("\n" + "="*60)
        print("АНАЛИЗ ПО ПОДРАЗДЕЛЕНИЯМ")
        print("="*60)
        
        forecast_data = self.df[(self.df['year'] == 2026) & (self.df['month'] >= 9)]
        
        subdivision_analysis = forecast_data.groupby('subdivision').agg({
            'revenue_total': 'sum',
            'traffic_total': 'sum',
            'transacitons_total': 'sum'
        }).sort_values('revenue_total', ascending=False)
        
        print("Подразделения по прогнозируемой выручке:")
        for subdivision, row in subdivision_analysis.iterrows():
            print(f"  {subdivision}: {row['revenue_total']:,.0f}")
        
        return subdivision_analysis
    
    def create_visualizations(self, monthly_data, historical, forecast):
        """Создание визуализаций"""
        print("\nСоздание графиков...")
        
        # Создаем фигуру с несколькими графиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ прогноза маркетинговых данных', fontsize=16, fontweight='bold')
        
        # График 1: Выручка по месяцам
        axes[0, 0].plot(historical['date'], historical['revenue_total'], 
                       'b-', label='Исторические данные', linewidth=2)
        axes[0, 0].plot(forecast['date'], forecast['revenue_total'], 
                       'r--', label='Прогноз', linewidth=2)
        axes[0, 0].set_title('Выручка по месяцам')
        axes[0, 0].set_ylabel('Выручка')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Трафик по месяцам
        axes[0, 1].plot(historical['date'], historical['traffic_total'], 
                       'b-', label='Исторические данные', linewidth=2)
        axes[0, 1].plot(forecast['date'], forecast['traffic_total'], 
                       'r--', label='Прогноз', linewidth=2)
        axes[0, 1].set_title('Трафик по месяцам')
        axes[0, 1].set_ylabel('Трафик')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Топ-10 категорий по выручке
        forecast_data = self.df[(self.df['year'] == 2026) & (self.df['month'] >= 9)]
        top_categories = forecast_data.groupby('category')['revenue_total'].sum().sort_values(ascending=False).head(10)
        
        axes[1, 0].barh(range(len(top_categories)), top_categories.values)
        axes[1, 0].set_yticks(range(len(top_categories)))
        axes[1, 0].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat 
                                   for cat in top_categories.index])
        axes[1, 0].set_title('Топ-10 категорий по прогнозируемой выручке')
        axes[1, 0].set_xlabel('Выручка')
        
        # График 4: Подразделения по выручке
        subdivision_revenue = forecast_data.groupby('subdivision')['revenue_total'].sum().sort_values(ascending=False)
        
        axes[1, 1].pie(subdivision_revenue.values, labels=subdivision_revenue.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Распределение выручки по подразделениям (прогноз)')
        
        plt.tight_layout()
        plt.savefig('forecast_analysis.png', dpi=300, bbox_inches='tight')
        print("Графики сохранены в файл: forecast_analysis.png")
        
        return fig
    
    def generate_budget_recommendations(self):
        """Генерация рекомендаций по бюджету"""
        print("\n" + "="*60)
        print("РЕКОМЕНДАЦИИ ПО БЮДЖЕТУ НА 2026 ГОД")
        print("="*60)
        
        # Анализируем прогноз на последние месяцы 2026 года
        forecast_data = self.df[(self.df['year'] == 2026) & (self.df['month'] >= 9)]
        
        # Средние значения за прогнозный период
        avg_revenue = forecast_data['revenue_total'].sum() / 4  # 4 месяца
        avg_traffic = forecast_data['traffic_total'].sum() / 4
        avg_ads_cost = forecast_data['ads_cost'].sum() / 4
        avg_mar_cost = forecast_data['mar_cost'].sum() / 4
        
        print(f"Средние показатели за прогнозный период (сентябрь-декабрь 2026):")
        print(f"  Выручка: {avg_revenue:,.0f} руб/месяц")
        print(f"  Трафик: {avg_traffic:,.0f} посетителей/месяц")
        print(f"  Расходы на рекламу: {avg_ads_cost:,.0f} руб/месяц")
        print(f"  Маркетинговые расходы: {avg_mar_cost:,.0f} руб/месяц")
        
        # Прогноз на 2027 год
        annual_revenue_2027 = avg_revenue * 12
        annual_traffic_2027 = avg_traffic * 12
        annual_ads_budget_2027 = avg_ads_cost * 12
        annual_mar_budget_2027 = avg_mar_cost * 12
        
        print(f"\nРекомендуемый бюджет на 2026 год:")
        print(f"  Прогнозируемая выручка: {annual_revenue_2027:,.0f} руб")
        print(f"  Прогнозируемый трафик: {annual_traffic_2027:,.0f} посетителей")
        print(f"  Бюджет на рекламу: {annual_ads_budget_2027:,.0f} руб")
        print(f"  Маркетинговый бюджет: {annual_mar_budget_2027:,.0f} руб")
        print(f"  Общий маркетинговый бюджет: {annual_ads_budget_2027 + annual_mar_budget_2027:,.0f} руб")
        
        # ROI анализ
        roi = (annual_revenue_2027 / (annual_ads_budget_2027 + annual_mar_budget_2027)) * 100
        print(f"  Прогнозируемый ROI: {roi:.1f}%")
        
        return {
            'annual_revenue': annual_revenue_2027,
            'annual_traffic': annual_traffic_2027,
            'annual_ads_budget': annual_ads_budget_2027,
            'annual_mar_budget': annual_mar_budget_2027,
            'roi': roi
        }
    
    def save_analysis_report(self, category_analysis, subdivision_analysis, budget_data):
        """Сохранение отчета анализа"""
        print("\nСохранение отчета анализа...")
        
        with open('forecast_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО АНАЛИЗУ ПРОГНОЗА МАРКЕТИНГОВЫХ ДАННЫХ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Дата создания отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("РЕКОМЕНДАЦИИ ПО БЮДЖЕТУ НА 2026 ГОД:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Прогнозируемая выручка: {budget_data['annual_revenue']:,.0f} руб\n")
            f.write(f"Прогнозируемый трафик: {budget_data['annual_traffic']:,.0f} посетителей\n")
            f.write(f"Бюджет на рекламу: {budget_data['annual_ads_budget']:,.0f} руб\n")
            f.write(f"Маркетинговый бюджет: {budget_data['annual_mar_budget']:,.0f} руб\n")
            f.write(f"Общий маркетинговый бюджет: {budget_data['annual_ads_budget'] + budget_data['annual_mar_budget']:,.0f} руб\n")
            f.write(f"Прогнозируемый ROI: {budget_data['roi']:.1f}%\n\n")
            
            f.write("ТОП-10 КАТЕГОРИЙ ПО ПРОГНОЗИРУЕМОЙ ВЫРУЧКЕ:\n")
            f.write("-" * 50 + "\n")
            for i, (category, row) in enumerate(category_analysis.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {category}: {row['revenue_total']:,.0f} руб\n")
            
            f.write("\nПОДРАЗДЕЛЕНИЯ ПО ПРОГНОЗИРУЕМОЙ ВЫРУЧКЕ:\n")
            f.write("-" * 45 + "\n")
            for subdivision, row in subdivision_analysis.iterrows():
                f.write(f"{subdivision}: {row['revenue_total']:,.0f} руб\n")
        
        print("Отчет сохранен в файл: forecast_analysis_report.txt")

def main():
    """Основная функция анализа"""
    print("Анализ качества прогноза маркетинговых данных")
    print("="*60)
    
    # Инициализация анализатора
    analyzer = ForecastAnalyzer('Marketing_Budget_Forecast_Extended.csv')
    
    # Загрузка данных
    analyzer.load_data()
    
    # Анализ трендов
    monthly_data, historical, forecast = analyzer.analyze_trends()
    
    # Анализ по категориям
    category_analysis = analyzer.analyze_categories()
    
    # Анализ по подразделениям
    subdivision_analysis = analyzer.analyze_subdivisions()
    
    # Создание визуализаций
    analyzer.create_visualizations(monthly_data, historical, forecast)
    
    # Генерация рекомендаций по бюджету
    budget_data = analyzer.generate_budget_recommendations()
    
    # Сохранение отчета
    analyzer.save_analysis_report(category_analysis, subdivision_analysis, budget_data)
    
    print("\nАнализ завершен успешно!")
    print("Созданные файлы:")
    print("  - forecast_analysis.png (графики)")
    print("  - forecast_analysis_report.txt (отчет)")

if __name__ == "__main__":
    main()
