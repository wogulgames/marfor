# Паттерны кода и примеры для сводных таблиц MARFOR

## 🔄 Правильные паттерны кода

### 1. Создание иерархической структуры

```javascript
// ✅ ПРАВИЛЬНО: Создаем все уровни агрегации
createHierarchicalRows() {
    const hierarchicalRows = new Map();
    
    // Собираем все уникальные комбинации
    const allKeys = new Set();
    this.rowGroups.forEach((rowGroup, rowKey) => {
        const keyParts = rowKey.split('|');
        
        // Создаем ключи для всех уровней
        for (let i = 1; i <= keyParts.length; i++) {
            const partialKey = keyParts.slice(0, i).join('|');
            allKeys.add(partialKey);
        }
    });
    
    // Создаем записи для всех ключей
    allKeys.forEach(key => {
        const keyParts = key.split('|');
        const level = keyParts.length - 1;
        const isAggregated = level < this.visibleTimeFields.length - 1;
        
        hierarchicalRows.set(key, {
            isAggregated,
            level,
            fields: this.createFieldsFromKey(key)
        });
    });
    
    return hierarchicalRows;
}
```

### 2. Правильная интерпретация collapsedRows

```javascript
// ✅ ПРАВИЛЬНО: collapsedRows содержит РАСКРЫТЫЕ элементы
isRowVisible(rowKey, collapsedRows) {
    const rowFields = rowKey.split('|');
    
    // Проверяем каждый уровень родительских элементов
    for (let level = 0; level < rowFields.length - 1; level++) {
        const parentKey = rowFields.slice(0, level + 1).join('|');
        
        // Если родитель НЕ в collapsedRows, значит он свернут
        if (!collapsedRows.has(parentKey)) {
            return false; // Скрываем дочерние элементы
        }
    }
    
    return true; // Все родительские элементы развернуты
}
```

### 3. Создание множественных кнопок коллапса

```javascript
// ✅ ПРАВИЛЬНО: Добавляем кнопки для всех родительских уровней
addRowsRecursively(currentKey, level) {
    const childRows = Array.from(hierarchicalRows.entries())
        .filter(([key, rowData]) => {
            // Логика фильтрации дочерних элементов
            return key.startsWith(currentKey + '|') && 
                   key.split('|').length === currentKey.split('|').length + 1;
        });
    
    childRows.forEach(([childKey, childData], index) => {
        if (index === 0) { // Только для первой дочерней строки
            // Создаем массив кнопок
            if (!childData.collapseButtons) {
                childData.collapseButtons = [];
            }
            
            // Добавляем кнопку для текущего уровня
            childData.collapseButtons.push({
                collapseKey: currentKey,
                collapseIcon: '-',
                level: level
            });
            
            // Добавляем кнопки для всех родительских уровней
            const childKeyParts = childKey.split('|');
            for (let parentLevel = 0; parentLevel < level; parentLevel++) {
                const parentKey = childKeyParts.slice(0, parentLevel + 1).join('|');
                
                if (hierarchicalRows.has(parentKey) && collapsedRows.has(parentKey)) {
                    childData.collapseButtons.push({
                        collapseKey: parentKey,
                        collapseIcon: '-',
                        level: parentLevel
                    });
                }
            }
        }
        
        // Рекурсивно обрабатываем дочерние элементы
        addRowsRecursively(childKey, level + 1);
    });
}
```

### 4. Рендеринг кнопок в правильных столбцах

```javascript
// ✅ ПРАВИЛЬНО: Кнопки в столбцах соответствующих уровней
visibleTimeFields.forEach((rowField, index) => {
    let cellContent = rowFields[rowField.name] || '';
    
    // Кнопка + для свернутых агрегированных строк
    if (rowData.isAggregated && index === rowData.level) {
        const collapseKey = rowKey;
        const isCollapsed = !collapsedRows.has(collapseKey);
        const collapseIcon = '+';
        
        cellContent = `<button onclick="toggleRowCollapse('${collapseKey}')">${collapseIcon}</button>${cellContent}`;
    }
    // Кнопки - для дочерних строк
    else if (rowData.collapseButtons) {
        const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
        
        if (buttonForLevel) {
            const collapseIcon = buttonForLevel.collapseIcon;
            const collapseKey = buttonForLevel.collapseKey;
            
            cellContent = `<button onclick="toggleRowCollapse('${collapseKey}')">${collapseIcon}</button>${cellContent}`;
        }
    }
    
    html += `<td>${cellContent}</td>`;
});
```

## ❌ Неправильные паттерны (избегайте!)

### 1. Неправильная интерпретация collapsedRows

```javascript
// ❌ НЕПРАВИЛЬНО: Путаница в логике
const isCollapsed = collapsedRows.has(rowKey); // Это означает РАСКРЫТО!
if (isCollapsed) {
    // Логика для свернутого состояния
} else {
    // Логика для раскрытого состояния
}
```

### 2. Создание только детальных строк

```javascript
// ❌ НЕПРАВИЛЬНО: Только lowest-level данные
createHierarchicalRows() {
    const hierarchicalRows = new Map();
    
    // Создаем только детальные строки
    this.rowGroups.forEach((rowGroup, rowKey) => {
        hierarchicalRows.set(rowKey, {
            isAggregated: false,
            level: rowKey.split('|').length - 1,
            fields: this.createFieldsFromKey(rowKey)
        });
    });
    
    return hierarchicalRows; // Отсутствуют родительские уровни!
}
```

### 3. Кнопки только в первом столбце

```javascript
// ❌ НЕПРАВИЛЬНО: Все кнопки - в первом столбце
if (shouldShowCollapseButton && index === 0) {
    // Кнопка - всегда в первом столбце
    cellContent = `<button>${collapseIcon}</button>${cellContent}`;
}
```

### 4. Отсутствие рекурсии в обработке детей

```javascript
// ❌ НЕПРАВИЛЬНО: Не обрабатываем вложенные уровни
childRows.forEach(([childKey, childData], index) => {
    // Только добавляем в массив, но не обрабатываем рекурсивно
    sortedRows.push([childKey, childData]);
    // НЕТ: addRowsRecursively(childKey, level + 1);
});
```

## 🧪 Паттерны отладки

### 1. Проверка структуры данных

```javascript
console.log('=== ОТЛАДКА СТРУКТУРЫ ===');
console.log('hierarchicalRows size:', hierarchicalRows.size);
console.log('collapsedRows size:', collapsedRows.size);
console.log('sortedRows length:', sortedRows.length);

console.log('hierarchicalRows keys:', Array.from(hierarchicalRows.keys()));
console.log('collapsedRows keys:', Array.from(collapsedRows));
```

### 2. Отладка логики кнопок

```javascript
console.log('=== ОТЛАДКА КНОПОК ===');
console.log('rowKey:', rowKey);
console.log('rowData.isAggregated:', rowData.isAggregated);
console.log('rowData.level:', rowData.level);
console.log('rowData.collapseButtons:', rowData.collapseButtons);
console.log('index:', index);
```

### 3. Отладка видимости строк

```javascript
console.log('=== ОТЛАДКА ВИДИМОСТИ ===');
console.log('rowKey:', rowKey);
console.log('isRowVisible result:', this.isRowVisible(rowKey, collapsedRows));

const rowFields = rowKey.split('|');
for (let level = 0; level < rowFields.length - 1; level++) {
    const parentKey = rowFields.slice(0, level + 1).join('|');
    const parentExpanded = collapsedRows.has(parentKey);
    console.log(`Parent ${parentKey} expanded: ${parentExpanded}`);
}
```

## 🔍 Типичные проблемы и их решения

### Проблема: "Плюсики не появляются"

**Диагностика:**
```javascript
// Проверяем hasChildrenForLevel
const hasChildren = this.hasChildrenForLevel(rowKeys, fieldName, fieldValue, fieldIndex, visibleTimeFields);
console.log(`hasChildrenForLevel: ${fieldName}=${fieldValue} -> ${hasChildren}`);
```

**Решение:**
```javascript
// Убедитесь, что созданы агрегированные строки
// Убедитесь, что rowKeys содержит родительские уровни
// Проверьте логику в hasChildrenForLevel
```

### Проблема: "Кнопки - не появляются"

**Диагностика:**
```javascript
console.log('collapseButtons:', rowData.collapseButtons);
console.log('buttonForLevel:', buttonForLevel);
console.log('index vs collapseButtonLevel:', index, rowData.collapseButtonLevel);
```

**Решение:**
```javascript
// Убедитесь, что collapseButtons создается в addRowsRecursively
// Проверьте, что level правильно передается
// Убедитесь, что логика поиска кнопки для уровня работает
```

### Проблема: "Таблица показывает только Total"

**Диагностика:**
```javascript
console.log('sortedRows:', sortedRows);
console.log('isRowVisible results:', sortedRows.map(([key, data]) => 
    `${key}: ${this.isRowVisible(key, collapsedRows)}`
));
```

**Решение:**
```javascript
// Проверьте логику в createHierarchicalSorting
// Убедитесь, что не все строки фильтруются как невидимые
// Проверьте интерпретацию collapsedRows
```

## 📝 Чек-лист для разработки

- [ ] Создана полная иерархическая структура (все уровни агрегации)
- [ ] Правильно интерпретируется `collapsedRows` (содержит раскрытые элементы)
- [ ] Кнопки `+` появляются для свернутых агрегированных строк
- [ ] Кнопки `-` создаются для всех родительских уровней
- [ ] Кнопки размещаются в правильных столбцах
- [ ] Рекурсивная обработка всех уровней иерархии
- [ ] Правильная логика видимости строк
- [ ] Отладочные логи для диагностики проблем

---
*Создано в версии 2.7.14 - Декабрь 2024*
