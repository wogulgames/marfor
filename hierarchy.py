"""
Модуль для иерархического прогнозирования (Hierarchical Forecasting).

Реализует методы согласования прогнозов (reconciliation):
- Bottom-Up: прогнозы с нижнего уровня агрегируются вверх
- Top-Down (Proportional): прогноз верхнего уровня распределяется по историческим долям
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class HierarchyReconciler:
    """Класс для согласования иерархических прогнозов"""
    
    def __init__(self, hierarchy_cols, metric):
        """
        Инициализация
        
        Args:
            hierarchy_cols: список колонок иерархии в порядке от верхнего к нижнему
                           например: ['region_to', 'subdivision', 'category']
            metric: название целевой метрики
        """
        self.hierarchy_cols = hierarchy_cols
        self.metric = metric
        self.hierarchy_levels = len(hierarchy_cols)
        
        print(f"\n🏗️ === ИЕРАРХИЧЕСКАЯ МОДЕЛЬ ===", flush=True)
        print(f"   Уровни иерархии ({len(hierarchy_cols)}): {' > '.join(hierarchy_cols)}", flush=True)
        print(f"   Метрика: {metric}", flush=True)
    
    def build_hierarchy_tree(self, df):
        """
        Строит дерево иерархии из данных
        
        Returns:
            dict: {уровень: список уникальных комбинаций}
        """
        hierarchy_tree = {}
        
        for level in range(len(self.hierarchy_cols) + 1):
            if level == 0:
                # Верхний уровень (total)
                hierarchy_tree[level] = [('TOTAL',)]
            else:
                # Уникальные комбинации до этого уровня
                cols_at_level = self.hierarchy_cols[:level]
                unique_combos = df[cols_at_level].drop_duplicates().values.tolist()
                hierarchy_tree[level] = [tuple(combo) for combo in unique_combos]
        
        print(f"\n   📊 Структура иерархии:", flush=True)
        for level, combos in hierarchy_tree.items():
            if level == 0:
                print(f"      Уровень {level} (Total): 1 элемент", flush=True)
            else:
                print(f"      Уровень {level} ({self.hierarchy_cols[:level]}): {len(combos)} элементов", flush=True)
        
        return hierarchy_tree
    
    def get_historical_proportions(self, df, time_col='month', year_col='year'):
        """
        Вычисляет исторические доли для top-down распределения
        
        Returns:
            dict: {parent_key: {child_key: proportion}}
        """
        proportions = {}
        
        # Для каждого уровня (кроме нижнего)
        for level in range(len(self.hierarchy_cols)):
            parent_cols = self.hierarchy_cols[:level] if level > 0 else []
            child_cols = self.hierarchy_cols[:level + 1]
            
            # Группируем по родительскому уровню
            if parent_cols:
                parent_totals = df.groupby(parent_cols)[self.metric].sum()
            else:
                parent_totals = pd.Series({('TOTAL',): df[self.metric].sum()})
            
            # Группируем по дочернему уровню
            child_totals = df.groupby(child_cols)[self.metric].sum()
            
            # Вычисляем доли
            for child_key, child_value in child_totals.items():
                if not isinstance(child_key, tuple):
                    child_key = (child_key,)
                
                parent_key = child_key[:-1] if len(child_key) > 1 else ('TOTAL',)
                parent_value = parent_totals.get(parent_key, 0)
                
                if parent_value > 0:
                    proportion = child_value / parent_value
                else:
                    proportion = 0
                
                if parent_key not in proportions:
                    proportions[parent_key] = {}
                
                proportions[parent_key][child_key] = proportion
        
        print(f"   ✅ Вычислены исторические пропорции для {len(proportions)} родительских узлов", flush=True)
        
        return proportions
    
    def reconcile_bottom_up(self, forecasts_df):
        """
        Согласование методом Bottom-Up
        
        Прогнозы нижнего уровня агрегируются вверх по иерархии
        
        Args:
            forecasts_df: DataFrame с прогнозами нижнего уровня
                         должен содержать все колонки иерархии + метрику
        
        Returns:
            DataFrame с прогнозами на всех уровнях
        """
        print(f"\n   🔼 Bottom-Up: агрегация прогнозов вверх по иерархии", flush=True)
        
        all_forecasts = []
        
        # Нижний уровень - используем как есть
        bottom_level_forecasts = forecasts_df.copy()
        all_forecasts.append(bottom_level_forecasts)
        
        # Агрегируем вверх по уровням
        for level in range(len(self.hierarchy_cols) - 1, -1, -1):
            if level == 0:
                # Верхний уровень (total)
                time_cols = [col for col in forecasts_df.columns if col in ['year', 'month', 'Halfyear', 'Quarter', 'period']]
                if time_cols:
                    level_forecast = forecasts_df.groupby(time_cols, dropna=False)[self.metric].sum().reset_index()
                    for col in self.hierarchy_cols:
                        level_forecast[col] = 'TOTAL'
                    all_forecasts.append(level_forecast)
            else:
                # Промежуточные уровни
                group_cols = self.hierarchy_cols[:level]
                time_cols = [col for col in forecasts_df.columns if col in ['year', 'month', 'Halfyear', 'Quarter', 'period']]
                group_by = time_cols + group_cols
                
                level_forecast = forecasts_df.groupby(group_by, dropna=False)[self.metric].sum().reset_index()
                
                # Добавляем пропущенные колонки иерархии
                for col in self.hierarchy_cols[level:]:
                    if col not in level_forecast.columns:
                        level_forecast[col] = 'AGGREGATED'
                
                all_forecasts.append(level_forecast)
        
        # Объединяем все уровни
        result = pd.concat(all_forecasts, ignore_index=True)
        
        print(f"   ✅ Создано прогнозов на всех уровнях: {len(result)} строк", flush=True)
        
        return result
    
    def reconcile_top_down(self, top_forecast, proportions, time_periods):
        """
        Согласование методом Top-Down (Proportional)
        
        Прогноз верхнего уровня распределяется вниз по историческим долям
        
        Args:
            top_forecast: прогноз для верхнего уровня (total)
                         dict: {period: value}
            proportions: исторические доли из get_historical_proportions()
            time_periods: список периодов для прогноза
        
        Returns:
            DataFrame с прогнозами на всех уровнях
        """
        print(f"\n   🔽 Top-Down: распределение прогноза по историческим долям", flush=True)
        
        all_forecasts = []
        
        # Для каждого периода
        for period_info in time_periods:
            period_key = f"{period_info['year']}-{period_info['month']:02d}"
            top_value = top_forecast.get(period_key, 0)
            
            # Распределяем по уровням
            self._distribute_forecast(
                parent_key=('TOTAL',),
                parent_value=top_value,
                proportions=proportions,
                period_info=period_info,
                level=0,
                forecasts_list=all_forecasts
            )
        
        result = pd.DataFrame(all_forecasts)
        
        print(f"   ✅ Создано прогнозов на всех уровнях: {len(result)} строк", flush=True)
        
        return result
    
    def _distribute_forecast(self, parent_key, parent_value, proportions, period_info, level, forecasts_list):
        """
        Рекурсивно распределяет прогноз по дочерним узлам
        """
        # Если достигли конца иерархии
        if level >= len(self.hierarchy_cols):
            return
        
        # Получаем дочерние узлы и их доли
        children = proportions.get(parent_key, {})
        
        if not children:
            # Нет дочерних узлов - это лист дерева
            return
        
        # Распределяем прогноз по дочерним узлам
        for child_key, proportion in children.items():
            child_value = parent_value * proportion
            
            # Создаем строку прогноза
            forecast_row = {
                'year': period_info['year'],
                'month': period_info['month'],
                'Halfyear': period_info.get('halfyear', 'H1' if period_info['month'] <= 6 else 'H2'),
                'Quarter': period_info.get('quarter', f'Q{(period_info["month"]-1)//3 + 1}'),
                self.metric: child_value
            }
            
            # Добавляем значения иерархии
            for i, col_name in enumerate(self.hierarchy_cols[:len(child_key)]):
                forecast_row[col_name] = child_key[i]
            
            # Заполняем остальные уровни как 'AGGREGATED'
            for col_name in self.hierarchy_cols[len(child_key):]:
                forecast_row[col_name] = 'AGGREGATED'
            
            forecasts_list.append(forecast_row)
            
            # Рекурсивно распределяем дальше вниз
            self._distribute_forecast(
                parent_key=child_key,
                parent_value=child_value,
                proportions=proportions,
                period_info=period_info,
                level=level + 1,
                forecasts_list=forecasts_list
            )

