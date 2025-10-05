# Быстрая справка по сводным таблицам MARFOR

## 🚨 КРИТИЧЕСКИ ВАЖНО

### `collapsedRows` - это Set РАСКРЫТЫХ элементов!
```javascript
// ✅ ПРАВИЛЬНО
const isExpanded = collapsedRows.has(rowKey);
const isCollapsed = !isExpanded;

// ❌ НЕПРАВИЛЬНО  
const isCollapsed = collapsedRows.has(rowKey); // Это РАСКРЫТО!
```

## 🎯 Основные принципы

### 1. Структура данных
- **Все уровни агрегации** должны быть в `hierarchicalRows`
- **Агрегированные строки** имеют `isAggregated: true`
- **Детальные строки** имеют `isAggregated: false`

### 2. Логика кнопок
- **Кнопки `+`**: для свернутых агрегированных строк
- **Кнопки `-`**: для первых дочерних элементов раскрытых родителей
- **Множественные кнопки `-`**: каждая для своего уровня

### 3. Размещение кнопок
- **Кнопка `+`**: в столбце `rowData.level`
- **Кнопка `-`**: в столбце `button.level`

## 🔧 Ключевые методы

| Метод | Назначение |
|-------|------------|
| `createHierarchicalRows()` | Создает все уровни агрегации |
| `createHierarchicalSorting()` | Определяет порядок и видимость строк |
| `createRowsHTML()` | Рендерит HTML с кнопками |
| `isRowVisible()` | Проверяет видимость строки |

## 🐛 Типичные ошибки

1. **Неправильная интерпретация `collapsedRows`**
2. **Отсутствие агрегированных строк** в `hierarchicalRows`
3. **Кнопки `-` только в первом столбце**
4. **Отсутствие рекурсии** в `addRowsRecursively`
5. **Слишком агрессивная фильтрация** в `createHierarchicalSorting`

## 🧪 Отладка

```javascript
// Проверка структуры
console.log('hierarchicalRows:', Array.from(hierarchicalRows.keys()));
console.log('collapsedRows:', Array.from(collapsedRows));

// Проверка логики кнопок
console.log('rowData.collapseButtons:', rowData.collapseButtons);
console.log('buttonForLevel:', buttonForLevel);

// Проверка видимости
console.log('isRowVisible:', this.isRowVisible(rowKey, collapsedRows));
```

## 📋 Чек-лист

- [ ] Все уровни в `hierarchicalRows`
- [ ] Правильная интерпретация `collapsedRows`
- [ ] Кнопки в правильных столбцах
- [ ] Рекурсивная обработка детей
- [ ] Отладочные логи добавлены

---
*Версия 2.7.14 - Декабрь 2024*
