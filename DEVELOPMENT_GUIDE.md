# Руководство по разработке сводных таблиц MARFOR

## 📋 Обзор проекта

MARFOR - это веб-приложение для прогнозирования маркетинговых данных с каскадной моделью Random Forest. Основная функциональность включает создание интерактивных сводных таблиц с иерархическим коллапсом/раскрытием строк.

**Версия:** 2.7.14  
**Основные файлы:**
- `working_web_app.py` - Flask backend
- `static/pivot_table_system.js` - JavaScript система сводных таблиц
- `templates/` - HTML шаблоны

## 🎯 Ключевая функциональность

### Сводная таблица с иерархическим коллапсом
- **Временные ряды**: Year → Halfyear → Quarter → Month
- **Кнопки `+`**: для свернутых родительских элементов
- **Кнопки `-`**: для быстрого схлопывания до нужного уровня
- **Множественные кнопки `-`**: каждая кнопка схлопывает до своего уровня

## ⚠️ КРИТИЧЕСКИ ВАЖНО: Типичные ошибки и их решения

### 1. Проблема с интерпретацией `collapsedRows`

**❌ ОШИБКА:** Неправильная интерпретация состояния `collapsedRows`

```javascript
// НЕПРАВИЛЬНО - путаница в логике
const isCollapsed = collapsedRows.has(rowKey); // Это означает РАСКРЫТО!
```

**✅ ПРАВИЛЬНО:**
```javascript
// collapsedRows содержит РАСКРЫТЫЕ элементы
const isExpanded = collapsedRows.has(rowKey);
const isCollapsed = !isExpanded;
```

**Объяснение:** `collapsedRows` - это Set, который содержит ключи **раскрытых** строк, а не свернутых!

### 2. Проблема с `hasChildrenForLevel` - всегда возвращает `false`

**❌ ОШИБКА:** Логика поиска детей работает только с lowest-level данными

```javascript
// Проблема: rowKeys содержит только детальные строки типа "2024|H1|Q1|1"
// Но мы ищем детей для "2024", которых нет в rowKeys
```

**✅ РЕШЕНИЕ:** Создавать агрегированные строки для всех уровней

```javascript
// В PivotData.createHierarchicalRows создаем строки для всех уровней:
// "2024", "2024|H1", "2024|H1|Q1", "2024|H1|Q1|1"
```

### 3. Проблема с дублированием строк при создании агрегированных данных

**❌ ОШИБКА:** Создание агрегированных строк приводит к артефактам

```javascript
// Создаем строки типа "2024" с пустыми значениями
// Это приводит к появлению строк с прочерками и нулями
```

**✅ РЕШЕНИЕ:** Использовать `createHierarchicalStructure` для явного создания структуры

```javascript
// Создаем структуру с флагами isAggregated
// Правильно рассчитываем суммы для агрегированных строк
```

### 4. Проблема с размещением кнопок коллапса

**❌ ОШИБКА:** Кнопка `-` всегда в первом столбце

```javascript
// Кнопка - всегда в index === 0
// Но должна быть в столбце соответствующего уровня
```

**✅ РЕШЕНИЕ:** Использовать `collapseButtonLevel` для правильного размещения

```javascript
// Кнопка - в столбце level родителя
if (index === rowData.collapseButtonLevel) {
    // Показать кнопку -
}
```

### 5. Проблема с множественными кнопками коллапса

**❌ ОШИБКА:** Только одна кнопка `-` на строку

```javascript
// childData.hasCollapseButton = true; // Только одна кнопка
```

**✅ РЕШЕНИЕ:** Массив кнопок для всех родительских уровней

```javascript
// childData.collapseButtons = [];
// Добавляем кнопки для всех раскрытых родителей
for (let parentLevel = 0; parentLevel < level; parentLevel++) {
    childData.collapseButtons.push({
        collapseKey: parentKey,
        collapseIcon: '-',
        level: parentLevel
    });
}
```

## 🔧 Архитектурные решения

### Структура данных

```javascript
// Иерархическая структура
hierarchicalRows = {
    "2024": { isAggregated: true, level: 0, fields: {...} },
    "2024|H1": { isAggregated: true, level: 1, fields: {...} },
    "2024|H1|Q1": { isAggregated: true, level: 2, fields: {...} },
    "2024|H1|Q1|1": { isAggregated: false, level: 3, fields: {...} }
}

// Состояние коллапса
collapsedRows = Set(["2024", "2024|H1"]) // Раскрытые элементы
```

### Логика кнопок

```javascript
// Кнопки + для свернутых элементов
if (rowData.isAggregated && index === rowData.level) {
    // Показать кнопку +
}

// Кнопки - для первых дочерних элементов
if (rowData.collapseButtons) {
    const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
    if (buttonForLevel) {
        // Показать кнопку - для этого уровня
    }
}
```

## 🐛 Типичные ошибки отладки

### 1. "Плюсики не появляются"

**Причины:**
- `hasChildrenForLevel` возвращает `false` для всех строк
- Отсутствуют агрегированные строки в `hierarchicalRows`
- Неправильная логика в `createRowsHTML`

**Отладка:**
```javascript
console.log('rowKeys:', rowKeys);
console.log('hierarchicalRows keys:', Array.from(hierarchicalRows.keys()));
console.log('hasChildrenForLevel result:', this.hasChildrenForLevel(...));
```

### 2. "Таблица показывает только Total"

**Причины:**
- Слишком агрессивная фильтрация в `createHierarchicalSorting`
- Неправильная интерпретация `collapsedRows`
- Все строки помечены как скрытые

**Отладка:**
```javascript
console.log('collapsedRows:', collapsedRows);
console.log('isRowVisible result:', this.isRowVisible(rowKey, collapsedRows));
console.log('sortedRows length:', sortedRows.length);
```

### 3. "Кнопки - не появляются"

**Причины:**
- `collapseButtons` не создается в `addRowsRecursively`
- Неправильная логика поиска кнопки для уровня
- Отсутствует `collapseButtonLevel`

**Отладка:**
```javascript
console.log('collapseButtons:', rowData.collapseButtons);
console.log('buttonForLevel:', buttonForLevel);
console.log('index vs level:', index, rowData.collapseButtonLevel);
```

## 🚀 Рекомендации для разработки

### 1. Всегда используйте отладочные логи

```javascript
console.log('Отладка:', { rowKey, isCollapsed, level, collapsedRows });
```

### 2. Проверяйте структуру данных

```javascript
console.log('hierarchicalRows size:', hierarchicalRows.size);
console.log('collapsedRows size:', collapsedRows.size);
console.log('sortedRows length:', sortedRows.length);
```

### 3. Тестируйте каждый уровень иерархии

- Level 0 (Year): кнопки + и -
- Level 1 (Halfyear): кнопки + и -
- Level 2 (Quarter): кнопки + и -
- Level 3 (Month): только данные

### 4. Помните о состояниях

- **Изначально**: все свернуто → кнопки +
- **После раскрытия**: родитель скрыт → дети видны с кнопками -
- **После схлопывания**: дети скрыты → родитель виден с кнопкой +

## 📚 Ключевые методы для понимания

### `PivotData.createHierarchicalRows()`
Создает иерархическую структуру данных со всеми уровнями агрегации.

### `PivotRenderer.createHierarchicalSorting()`
Определяет порядок отображения строк и их видимость.

### `PivotRenderer.createRowsHTML()`
Рендерит HTML с кнопками коллапса в правильных столбцах.

### `PivotRenderer.isRowVisible()`
Определяет, должна ли строка быть видимой на основе состояния коллапса.

## 🎯 Заключение

Основные проблемы связаны с:
1. **Неправильной интерпретацией** `collapsedRows`
2. **Отсутствием агрегированных строк** для всех уровней
3. **Неправильным размещением кнопок** коллапса
4. **Сложной логикой** множественных кнопок

**Главное правило:** Всегда создавайте полную иерархическую структуру данных и правильно интерпретируйте состояния коллапса!

---
*Создано в версии 2.7.14 - Декабрь 2024*
