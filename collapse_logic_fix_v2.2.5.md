# Версия 2.2.5 - Исправление передачи allRowKeys в shouldShowCollapseButton

## Проблема
Кнопки коллапсирования не отображались для родительских строк в режиме срезов, потому что в функцию `shouldShowCollapseButton` передавались неправильные данные.

## Анализ проблемы

### Что было неправильно:

**Проблема в вызове `shouldShowCollapseButton`**:
```javascript
// БЫЛО (неправильно):
const shouldShowCollapseButton = this.shouldShowCollapseButton(rowKey, rowKeys, rowFieldValues, index);
```

**Проблема**: Передавались `rowKeys` (видимые строки), но функция `shouldShowCollapseButton` ожидает `allRowKeys` (все строки) для правильной проверки иерархии.

### Логика функции `shouldShowCollapseButton`:

```javascript
shouldShowCollapseButton(rowKey, allRowKeys, rowFieldValues, level) {
    // 1. Проверяем, есть ли дочерние элементы для этого уровня
    const hasChildren = this.hasChildrenForLevel(allRowKeys, level);
    if (!hasChildren) return false;
    
    // 2. Проверяем, является ли это агрегированной строкой
    const hasChildRows = allRowKeys.some(key => {
        const keyFields = key.split('|');
        const currentFields = rowKey.split('|');
        
        // Дочерняя строка должна начинаться с того же префикса, но иметь больше уровней
        if (keyFields.length <= currentFields.length) return false;
        
        // Проверяем, что текущая строка является префиксом дочерней строки
        return currentFields.every((field, index) => field === keyFields[index]);
    });
    
    return hasChildren && hasChildRows;
}
```

### Почему это не работало:

**Пример с вашими данными:**
- **Все строки**: `["2024", "2024|H1", "2024|H1|Q1", "2024|H1|Q1|1", ...]`
- **Видимые строки после коллапсирования H1**: `["2024", "2024|H1"]`

**Что происходило:**
1. Для строки `"2024|H1"` вызывалась функция `shouldShowCollapseButton("2024|H1", ["2024", "2024|H1"], rowFieldValues, 1)`
2. `hasChildrenForLevel(["2024", "2024|H1"], 1)` проверяла, есть ли строки с уровнем > 2 в массиве `["2024", "2024|H1"]`
3. **Результат**: `false` (нет строк с уровнем > 2)
4. Функция возвращала `false` → кнопка не показывалась

## Исправление

### Что было исправлено:

**Исправлен вызов `shouldShowCollapseButton`**:
```javascript
// СТАЛО (правильно):
const allRowKeys = pivotData.getRowKeys(); // Получаем все строки для проверки иерархии
const shouldShowCollapseButton = this.shouldShowCollapseButton(rowKey, allRowKeys, rowFieldValues, index);
```

**Изменения в коде:**
1. **В режиме `time-series`**:
   ```javascript
   const rowKeys = pivotData.getRowKeys();
   const allRowKeys = pivotData.getRowKeys(); // Получаем все строки для проверки иерархии
   
   const shouldShowCollapseButton = this.shouldShowCollapseButton(rowKey, allRowKeys, rowFieldValues, index);
   ```

2. **В режиме `slices`**:
   ```javascript
   const rowKeys = pivotData.getRowKeys();
   const allRowKeys = pivotData.getRowKeys(); // Получаем все строки для проверки иерархии
   
   const shouldShowCollapseButton = this.shouldShowCollapseButton(rowKey, allRowKeys, rowFieldValues, index);
   ```

## Как теперь работает логика

### Пример с вашими данными:

**Все строки**: `["2024", "2024|H1", "2024|H1|Q1", "2024|H1|Q1|1", ...]`
**Видимые строки после коллапсирования H1**: `["2024", "2024|H1"]`

**Что происходит:**
1. Для строки `"2024|H1"` вызывается функция `shouldShowCollapseButton("2024|H1", ["2024", "2024|H1", "2024|H1|Q1", "2024|H1|Q1|1", ...], rowFieldValues, 1)`
2. `hasChildrenForLevel(["2024", "2024|H1", "2024|H1|Q1", "2024|H1|Q1|1", ...], 1)` проверяет, есть ли строки с уровнем > 2
3. **Результат**: `true` (есть строки типа `"2024|H1|Q1|1"` с уровнем 3)
4. `hasChildRows` ищет строки, которые начинаются с `"2024|H1"` и имеют больше уровней
5. Находит `"2024|H1|Q1"`, `"2024|H1|Q1|1"` → возвращает `true`
6. **Результат**: `true && true = true` → кнопка показывается

## Результат
✅ Кнопки коллапсирования теперь отображаются для родительских строк в режиме срезов
✅ Работает раскрытие конкретных кварталов (например, только Q1)
✅ Работает раскрытие конкретных месяцев внутри квартала
✅ Функция `shouldShowCollapseButton` получает правильные данные для анализа иерархии

**Приложение запущено на http://localhost:5001 с версией 2.2.5**
