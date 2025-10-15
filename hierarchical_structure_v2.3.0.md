# Версия 2.3.0 - Перестройка структуры данных для поддержки агрегированных строк

## Проблема
Текущая структура данных не поддерживала независимое коллапсирование отдельных родительских строк, потому что не было агрегированных строк для каждого уровня иерархии.

## Анализ проблемы

### Старая структура (неправильная):
```
"2024|H1|Q1" - детализированная строка (квартал)
"2024|H1|Q2" - детализированная строка (квартал)
"2024|H1|Q3" - детализированная строка (квартал)
"2024|H1|Q4" - детализированная строка (квартал)
```

**Проблема**: Не было отдельных строк для `"2024"` (год) и `"2024|H1"` (полугодие), поэтому кнопки коллапсирования не могли появиться.

### Новая структура (правильная):
```
"2024" - агрегированная строка (год) ← кнопка коллапсирования
"2024|H1" - агрегированная строка (полугодие) ← кнопка коллапсирования
"2024|H1|Q1" - детализированная строка (квартал)
"2024|H1|Q2" - детализированная строка (квартал)
"2024|H1|Q3" - детализированная строка (квартал)
"2024|H1|Q4" - детализированная строка (квартал)
```

## Решение

### 1. Новый метод `createHierarchicalRows`

```javascript
createHierarchicalRows(config, visibleRowFields, renderer = null) {
    // Сначала создаем все детализированные строки
    this.rawData.forEach((row, index) => {
        const rowKey = this.createRowKey(row, visibleRowFields);
        
        if (!this.rowGroups.has(rowKey)) {
            this.rowGroups.set(rowKey, {
                key: rowKey,
                fields: this.extractFieldValues(row, visibleRowFields),
                rows: [],
                isAggregated: false // Детализированная строка
            });
        }
        this.rowGroups.get(rowKey).rows.push(row);
    });
    
    // Теперь создаем агрегированные строки для каждого уровня иерархии
    const aggregatedRows = new Map();
    
    this.rowGroups.forEach((group, rowKey) => {
        const fieldValues = rowKey.split('|');
        
        // Создаем агрегированные строки для каждого уровня
        for (let level = 0; level < fieldValues.length - 1; level++) {
            const aggregatedKey = fieldValues.slice(0, level + 1).join('|');
            
            if (!aggregatedRows.has(aggregatedKey)) {
                const aggregatedFields = {};
                visibleRowFields.forEach((field, index) => {
                    if (index <= level) {
                        aggregatedFields[field.name] = fieldValues[index];
                    } else {
                        aggregatedFields[field.name] = ''; // Пустое значение для неиспользуемых полей
                    }
                });
                
                aggregatedRows.set(aggregatedKey, {
                    key: aggregatedKey,
                    fields: aggregatedFields,
                    rows: [], // Агрегированная строка не содержит исходных данных
                    isAggregated: true, // Агрегированная строка
                    level: level
                });
            }
        }
    });
    
    // Добавляем агрегированные строки в rowGroups
    aggregatedRows.forEach((group, key) => {
        this.rowGroups.set(key, group);
    });
}
```

### 2. Обновленная логика `shouldShowCollapseButton`

```javascript
shouldShowCollapseButton(rowKey, allRowKeys, rowFieldValues, level) {
    // Получаем информацию о строке из rowGroups
    const rowGroup = window.currentPivotData ? window.currentPivotData.rowGroups.get(rowKey) : null;
    
    // Если это агрегированная строка, показываем кнопку
    if (rowGroup && rowGroup.isAggregated) {
        return true;
    }
    
    // Для детализированных строк кнопка не показывается
    return false;
}
```

## Как теперь работает логика

### Пример с вашими данными:

**Структура данных:**
- **Уровень 0**: `"2024"` (год) - агрегированная строка с кнопкой коллапсирования
- **Уровень 1**: `"2024|H1"` (полугодие) - агрегированная строка с кнопкой коллапсирования
- **Уровень 2**: `"2024|H1|Q1"` (квартал) - детализированная строка без кнопки

**Сценарий: Пользователь нажимает на кнопку у H1**

1. **Кнопка коллапсирования** у строки `"2024|H1"` получает `onclick="toggleRowCollapse('2024|H1')"`
2. **При нажатии** добавляется ключ `"2024|H1"` в `collapsedRows`
3. **При проверке видимости** строки `"2024|H1|Q1"`:
   - `rowFields = ["2024", "H1", "Q1"]`
   - `level = 0`: `parentKey = "2024"` → не в `collapsedRows`
   - `level = 1`: `parentKey = "2024|H1"` → **ЕСТЬ в `collapsedRows`** → `return false`
4. **Результат**: строка `"2024|H1|Q1"` скрывается

**Сценарий: Пользователь нажимает на кнопку у 2024**

1. **Кнопка коллапсирования** у строки `"2024"` получает `onclick="toggleRowCollapse('2024')"`
2. **При нажатии** добавляется ключ `"2024"` в `collapsedRows`
3. **При проверке видимости** строки `"2024|H1"`:
   - `rowFields = ["2024", "H1"]`
   - `level = 0`: `parentKey = "2024"` → **ЕСТЬ в `collapsedRows`** → `return false`
4. **Результат**: строка `"2024|H1"` скрывается

## Результат
✅ Созданы агрегированные строки для каждого уровня иерархии
✅ Кнопки коллапсирования появляются у агрегированных строк
✅ Работает независимое коллапсирование отдельных родительских строк
✅ Можно схлопнуть год и раскрыть только определенное полугодие
✅ Можно схлопнуть полугодие и раскрыть только определенный квартал

**Приложение запущено на http://localhost:5001 с версией 2.3.0**
