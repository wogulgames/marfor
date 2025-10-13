// ========================================
// НОВАЯ СИСТЕМА СВОДНОЙ ТАБЛИЦЫ
// ========================================
// 🔧 ВЕРСИЯ КОДА: 2.5.2 - Улучшены кнопки коллапсирования и добавлено выделение раскрытых полей

console.log('Загружается новая система сводной таблицы...');

// Классы для сущностей сводной таблицы
class PivotField {
    constructor(name, label, type, level = 0) {
        this.name = name;
        this.label = label || name;
        this.type = type; // 'time', 'slice', 'metric'
        this.level = level;
        this.aggregation = 'sum'; // sum, avg, count, etc.
    }
}

// Класс для фильтров
class PivotFilter {
    constructor(fieldName, fieldType, fieldLabel) {
        this.fieldName = fieldName;
        this.fieldType = fieldType; // 'text', 'number', 'date'
        this.fieldLabel = fieldLabel;
        this.isActive = false;
        this.values = []; // для текстовых фильтров - выбранные значения
        this.minValue = null; // для числовых фильтров - минимальное значение
        this.maxValue = null; // для числовых фильтров - максимальное значение
        this.availableValues = []; // доступные значения для фильтра
    }
    
    // Получение уникальных значений из данных
    getAvailableValues(rawData) {
        const values = new Set();
        rawData.forEach(row => {
            if (row[this.fieldName] !== null && row[this.fieldName] !== undefined) {
                values.add(row[this.fieldName]);
            }
        });
        
        this.availableValues = Array.from(values);
        
        // Для числовых полей сортируем и устанавливаем min/max
        if (this.fieldType === 'number') {
            this.availableValues.sort((a, b) => a - b);
            this.minValue = this.availableValues[0];
            this.maxValue = this.availableValues[this.availableValues.length - 1];
        } else {
            // Для текстовых полей сортируем по алфавиту
            this.availableValues.sort();
        }
        
        return this.availableValues;
    }
    
    // Проверка, проходит ли строка через фильтр
    matches(row) {
        console.log(`🔍 matches вызван для ${this.fieldName}, isActive: ${this.isActive}`);
        if (!this.isActive) return true;
        
        const value = row[this.fieldName];
        console.log(`🔍 value: ${value}, fieldType: ${this.fieldType}`);
        if (value === null || value === undefined) return false;
        
        if (this.fieldType === 'text') {
            return this.values.includes(value);
        } else if (this.fieldType === 'number' || this.fieldType === 'numeric') {
            const result = value >= this.minValue && value <= this.maxValue;
            console.log(`🔍 Фильтр ${this.fieldName}: value=${value}, minValue=${this.minValue}, maxValue=${this.maxValue}, result=${result}`);
            return result;
        }
        
        return true;
    }
    
    // Сброс фильтра
    reset() {
        this.isActive = false;
        this.values = [];
        if (this.fieldType === 'number') {
            this.minValue = this.availableValues[0];
            this.maxValue = this.availableValues[this.availableValues.length - 1];
        }
    }
}

class PivotConfig {
    constructor() {
        this.rows = []; // Поля для строк
        this.columns = []; // Поля для столбцов  
        this.values = []; // Поля для значений
        this.filters = []; // Фильтры
        this.mode = 'normal'; // normal, time-series, slices, split-columns
        this.originalMode = ''; // Исходный режим для split-columns
        this.sortConfig = {
            field: null, // Поле для сортировки
            direction: 'asc', // 'asc' или 'desc'
            type: 'text' // 'text', 'number', 'date'
        };
    }
    
    setRows(fields) {
        this.rows = fields;
    }
    
    setColumns(fields) {
        this.columns = fields;
    }
    
    setValues(fields) {
        this.values = fields;
    }
    
    setMode(mode) {
        this.mode = mode;
    }
    
    setOriginalMode(originalMode) {
        this.originalMode = originalMode;
    }
    
    setSortConfig(field, direction, type) {
        this.sortConfig = {
            field: field,
            direction: direction,
            type: type
        };
    }
    
    toggleSort(field, type) {
        if (this.sortConfig.field === field) {
            // Переключаем направление сортировки
            this.sortConfig.direction = this.sortConfig.direction === 'asc' ? 'desc' : 'asc';
        } else {
            // Новое поле для сортировки
            this.sortConfig.field = field;
            this.sortConfig.direction = 'asc';
            this.sortConfig.type = type;
        }
    }
}

class PivotData {
    constructor(rawData) {
        this.rawData = rawData;
        this.processedData = null;
        this.rowGroups = new Map();
        this.columnGroups = new Map();
        this.crossTable = new Map();
    }
    
    process(config, renderer = null) {
        console.log('Обработка данных сводной таблицы с конфигурацией:', config);
        
        // Группируем данные по строкам и столбцам
        this.groupByRowsAndColumns(config, renderer);
        
        // Создаем перекрестную таблицу
        this.createCrossTable(config, renderer);
        
        return this;
    }
    
    groupByRowsAndColumns(config, renderer = null) {
        this.rowGroups.clear();
        this.columnGroups.clear();
        
        // Данные уже отфильтрованы на уровне клиента, используем их как есть
        console.log(`Обрабатываем данные: ${this.rawData.length} строк`);
        
        // Получаем поля для группировки строк в зависимости от режима
        let visibleRowFields;
        if (config.mode === 'slices' || (config.mode === 'split-columns' && config.originalMode === 'slices')) {
            // В режиме срезов или split-columns из срезов группируем по видимым полям срезов
            visibleRowFields = renderer ? renderer.getVisibleSliceFields(config) : config.rows;
        } else {
            // В других режимах используем видимые временные поля
            visibleRowFields = renderer ? renderer.getVisibleTimeFields(config) : config.rows;
        }
        
        console.log('Группировка строк по полям:', visibleRowFields.map(f => f.name));
        
        // Создаем иерархическую структуру строк (детализированные + агрегированные)
        this.createHierarchicalRows(config, visibleRowFields);
        
        // Группировка по столбцам
        this.rawData.forEach(row => {
            const colKey = this.createColumnKey(row, config.columns);
            if (!this.columnGroups.has(colKey)) {
                this.columnGroups.set(colKey, {
                    key: colKey,
                    fields: this.extractFieldValues(row, config.columns),
                    rows: []
                });
            }
            this.columnGroups.get(colKey).rows.push(row);
        });
        
        console.log('Группировка завершена:', {
            rowGroups: this.rowGroups.size,
            columnGroups: this.columnGroups.size,
            visibleRowFields: visibleRowFields.length
        });
    }
    
    createCrossTable(config, renderer = null) {
        this.crossTable.clear();
        
        // Получаем поля для создания ключей в зависимости от режима
        let visibleRowFields;
        if (config.mode === 'slices' || (config.mode === 'split-columns' && config.originalMode === 'slices')) {
            // В режиме срезов или split-columns из срезов группируем по видимым полям срезов
            visibleRowFields = renderer ? renderer.getVisibleSliceFields(config) : config.rows;
        } else {
            // В других режимах используем видимые временные поля
            visibleRowFields = renderer ? renderer.getVisibleTimeFields(config) : config.rows;
        }
        
        console.log('Создание перекрестной таблицы:', {
            rawDataLength: this.rawData.length,
            visibleRows: visibleRowFields.map(r => r.name),
            columns: config.columns.map(c => c.name),
            values: config.values.map(v => v.name)
        });
        
        // Создаем перекрестную таблицу: rowKey -> colKey -> aggregatedValue
        this.rawData.forEach((row, index) => {
            if (index < 3) { // Логируем первые 3 строки
                console.log(`Строка ${index}:`, row);
            }
            
            const rowKey = this.createRowKey(row, visibleRowFields);
            const colKey = this.createColumnKey(row, config.columns);
            
            if (!this.crossTable.has(rowKey)) {
                this.crossTable.set(rowKey, new Map());
            }
            
            if (!this.crossTable.get(rowKey).has(colKey)) {
                this.crossTable.get(rowKey).set(colKey, {});
            }
            
            // Агрегируем значения
            config.values.forEach(valueField => {
                const currentValue = this.crossTable.get(rowKey).get(colKey)[valueField.name] || 0;
                const rowValue = parseFloat(row[valueField.name]) || 0;
                
                if (index < 3) { // Логируем для первых строк
                    console.log(`  Метрика ${valueField.name}: ${row[valueField.name]} -> ${rowValue}`);
                }
                
                if (valueField.aggregation === 'sum') {
                    this.crossTable.get(rowKey).get(colKey)[valueField.name] = currentValue + rowValue;
                } else if (valueField.aggregation === 'avg') {
                    // Для среднего нужно хранить количество и сумму
                    if (!this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_count`]) {
                        this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_count`] = 0;
                        this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_sum`] = 0;
                    }
                    this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_count`]++;
                    this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_sum`] += rowValue;
                    this.crossTable.get(rowKey).get(colKey)[valueField.name] = 
                        this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_sum`] / 
                        this.crossTable.get(rowKey).get(colKey)[`${valueField.name}_count`];
                } else if (valueField.aggregation === 'count') {
                    this.crossTable.get(rowKey).get(colKey)[valueField.name] = currentValue + 1;
                }
            });
        });
        
        // Агрегированные значения не нужны - работаем только с детализированными строками
        
        console.log('Перекрестная таблица создана:', {
            rows: this.crossTable.size,
            totalCells: Array.from(this.crossTable.values()).reduce((sum, colMap) => sum + colMap.size, 0)
        });
    }
    
    createAggregatedValues(config, visibleRowFields) {
        console.log('Создание агрегированных значений...');
        
        // Создаем агрегированные значения для всех уровней
        this.rowGroups.forEach((group, rowKey) => {
            if (!group.isAggregated) return; // Пропускаем детализированные строки
            
            const keyParts = rowKey.split('|');
            const level = keyParts.length - 1;
            
            // Находим все дочерние строки для этого уровня
            const childRows = [];
            this.rowGroups.forEach((childGroup, childKey) => {
                const childParts = childKey.split('|');
                
                // Проверяем, является ли это дочерней строкой
                if (childParts.length === level + 2 && 
                    keyParts.every((part, index) => part === childParts[index])) {
                    childRows.push(childKey);
                }
            });
            
            // Создаем агрегированные значения для всех колонок
            config.columns.forEach(colField => {
                if (colField.role !== 'metric') return;
                
                const colKey = colField.name;
                
                // Инициализируем колонку для этой строки, если её нет
                if (!this.crossTable.has(rowKey)) {
                    this.crossTable.set(rowKey, new Map());
                }
                if (!this.crossTable.get(rowKey).has(colKey)) {
                    this.crossTable.get(rowKey).set(colKey, {});
                }
                
                // Суммируем значения из дочерних строк
                let aggregatedValue = 0;
                childRows.forEach(childKey => {
                    if (this.crossTable.has(childKey) && this.crossTable.get(childKey).has(colKey)) {
                        const childValue = this.crossTable.get(childKey).get(colKey)[colField.name] || 0;
                        aggregatedValue += childValue;
                    }
                });
                
                // Если нет дочерних строк, суммируем из детализированных данных
                if (childRows.length === 0) {
                    group.rows.forEach(row => {
                        const value = parseFloat(row[colField.name]) || 0;
                        aggregatedValue += value;
                    });
                }
                
                this.crossTable.get(rowKey).get(colKey)[colField.name] = aggregatedValue;
            });
        });
        
        console.log('Агрегированные значения созданы');
    }
    
    
    createRowKey(row, rowFields) {
        return rowFields.map(field => row[field.name] || '').join('|');
    }
    
    createColumnKey(row, columnFields) {
        return columnFields.map(field => row[field.name] || '').join('|');
    }
    
    extractFieldValues(row, fields) {
        const result = {};
        fields.forEach(field => {
            result[field.name] = row[field.name] || '';
        });
        return result;
    }
    
    createHierarchicalRows(config, visibleRowFields) {
        console.log('Создание иерархической структуры строк...');
        
        // Сначала создаем детализированные строки
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
        
        // Создаем агрегированные строки для всех уровней
        const aggregatedRows = new Map();
        
        this.rowGroups.forEach((group, detailedKey) => {
            const keyParts = detailedKey.split('|');
            
            // Создаем агрегированные ключи для всех уровней
            for (let level = 0; level < keyParts.length; level++) {
                const aggregatedKey = keyParts.slice(0, level + 1).join('|');
                
                if (!aggregatedRows.has(aggregatedKey)) {
                    aggregatedRows.set(aggregatedKey, {
                        key: aggregatedKey,
                        fields: keyParts.slice(0, level + 1),
                        rows: [],
                        isAggregated: true,
                        level: level
                    });
                }
                
                // Добавляем детализированные строки к агрегированной
                aggregatedRows.get(aggregatedKey).rows.push(...group.rows);
            }
        });
        
        // Добавляем агрегированные строки к rowGroups
        aggregatedRows.forEach((group, key) => {
            // Создаем правильные поля для агрегированной строки
            const fields = {};
            const keyParts = key.split('|');
            visibleRowFields.forEach((field, index) => {
                if (index < keyParts.length) {
                    fields[field.name] = keyParts[index];
                } else {
                    fields[field.name] = '';
                }
            });
            
            this.rowGroups.set(key, {
                ...group,
                fields: fields
            });
        });
        
        console.log('Иерархическая структура создана:', {
            totalRows: this.rowGroups.size,
            detailedRows: this.rowGroups.size
        });
        
        // Выводим все ключи для отладки
        console.log('Все ключи строк:', Array.from(this.rowGroups.keys()));
    }
    
    getRowKeys() {
        // Возвращаем ключи в том порядке, в котором они находятся в Map
        // (уже отсортированы методом sortData, если была применена сортировка)
        return Array.from(this.rowGroups.keys());
    }
    
    getColumnKeys() {
        return Array.from(this.columnGroups.keys()).sort();
    }
    
    // Сортировка данных по указанному полю
    sortData(sortConfig) {
        console.log('🔍 sortData вызван с конфигурацией:', sortConfig);
        
        if (!sortConfig.field) {
            console.log('❌ sortData: нет поля для сортировки, выходим');
            return;
        }
        
        console.log('=== НАЧАЛО СОРТИРОВКИ ДАННЫХ ===');
        console.log('Применяем сортировку:', sortConfig);
        console.log('Количество rowGroups:', this.rowGroups.size);
        console.log('Примеры ключей rowGroups ДО сортировки:', Array.from(this.rowGroups.keys()).slice(0, 10));
        
        const sortedRowKeys = Array.from(this.rowGroups.keys()).sort((a, b) => {
            const rowA = this.rowGroups.get(a);
            const rowB = this.rowGroups.get(b);
            
            let valueA, valueB;
            
            // Проверяем, является ли поле полем строки (dimension)
            const isDimensionField = rowA.fields.hasOwnProperty(sortConfig.field);
            
            // Проверяем, является ли это сортировкой по конкретному столбцу метрики (например, revenue_first_transactions_2024)
            // Это возможно только в режиме split-columns, где crossTable содержит данные
            let isColumnMetric = false;
            let colKey = null;
            let metricName = null;
            
            if (!isDimensionField && sortConfig.field.includes('_') && this.crossTable.size > 0) {
                // Пробуем разделить на метрику и ключ столбца
                const parts = sortConfig.field.split('_');
                const potentialColKey = parts[parts.length - 1];
                const potentialMetricName = parts.slice(0, -1).join('_');
                
                // Проверяем, существует ли такой столбец в crossTable
                const firstRowKey = this.crossTable.keys().next().value;
                if (firstRowKey && this.crossTable.get(firstRowKey)?.has(potentialColKey)) {
                    isColumnMetric = true;
                    colKey = potentialColKey;
                    metricName = potentialMetricName;
                }
            }
            
            if (isColumnMetric && sortConfig.type === 'number') {
                // Сортировка по конкретному столбцу метрики (например, revenue_first_transactions_belarus)
                // Для агрегированных строк нужно суммировать значения из всех дочерних элементов
                
                // Проверяем, является ли строка агрегированной
                const isAggregatedA = !this.crossTable.has(a);
                const isAggregatedB = !this.crossTable.has(b);
                
                if (isAggregatedA) {
                    // Суммируем значения из всех дочерних элементов
                    valueA = 0;
                    const allRowKeys = Array.from(this.rowGroups.keys());
                    allRowKeys.forEach(childKey => {
                        if (childKey.startsWith(a + '|')) {
                            valueA += this.crossTable.get(childKey)?.get(colKey)?.[metricName] || 0;
                        }
                    });
                } else {
                    valueA = this.crossTable.get(a)?.get(colKey)?.[metricName] || 0;
                }
                
                if (isAggregatedB) {
                    // Суммируем значения из всех дочерних элементов
                    valueB = 0;
                    const allRowKeys = Array.from(this.rowGroups.keys());
                    allRowKeys.forEach(childKey => {
                        if (childKey.startsWith(b + '|')) {
                            valueB += this.crossTable.get(childKey)?.get(colKey)?.[metricName] || 0;
                        }
                    });
                } else {
                    valueB = this.crossTable.get(b)?.get(colKey)?.[metricName] || 0;
                }
                
                console.log('Сортировка по столбцу метрики:', {
                    field: sortConfig.field, 
                    metricName, 
                    colKey, 
                    rowKeyA: a,
                    rowKeyB: b,
                    isAggregatedA,
                    isAggregatedB,
                    valueA, 
                    valueB
                });
            } else if (!isDimensionField && sortConfig.type === 'number') {
                // Это метрика - суммируем значения
                // В режиме split-columns используем crossTable, в обычном режиме - rowA.rows
                
                if (this.crossTable.size > 0) {
                    // Режим split-columns: суммируем значения из crossTable по всем столбцам
                    const columnKeys = Array.from(this.columnGroups.keys());
                    
                    // Для агрегированных строк суммируем значения из всех дочерних элементов
                    const isAggregatedA = !this.crossTable.has(a);
                    const isAggregatedB = !this.crossTable.has(b);
                    
                    valueA = 0;
                    valueB = 0;
                    
                    if (isAggregatedA) {
                        // Суммируем значения из всех дочерних элементов по всем столбцам
                        const allRowKeys = Array.from(this.rowGroups.keys());
                        allRowKeys.forEach(childKey => {
                            if (childKey.startsWith(a + '|')) {
                                columnKeys.forEach(colKey => {
                                    valueA += this.crossTable.get(childKey)?.get(colKey)?.[sortConfig.field] || 0;
                                });
                            }
                        });
                    } else {
                        // Обычная строка - суммируем по всем столбцам
                        columnKeys.forEach(colKey => {
                            valueA += this.crossTable.get(a)?.get(colKey)?.[sortConfig.field] || 0;
                        });
                    }
                    
                    if (isAggregatedB) {
                        // Суммируем значения из всех дочерних элементов по всем столбцам
                        const allRowKeys = Array.from(this.rowGroups.keys());
                        allRowKeys.forEach(childKey => {
                            if (childKey.startsWith(b + '|')) {
                                columnKeys.forEach(colKey => {
                                    valueB += this.crossTable.get(childKey)?.get(colKey)?.[sortConfig.field] || 0;
                                });
                            }
                        });
                    } else {
                        // Обычная строка - суммируем по всем столбцам
                        columnKeys.forEach(colKey => {
                            valueB += this.crossTable.get(b)?.get(colKey)?.[sortConfig.field] || 0;
                        });
                    }
                    
                    // Логируем для отладки
                    if (Math.random() < 0.05) {
                        console.log('Сортировка по метрике в split-columns (пример):', { 
                            field: sortConfig.field, 
                            rowKeyA: a,
                            rowKeyB: b,
                            isAggregatedA,
                            isAggregatedB,
                    valueA, 
                    valueB,
                            columnsCount: columnKeys.length
                });
                    }
                } else {
                    // Обычный режим: суммируем значения из rowA.rows
                let sumA = 0, sumB = 0;
                    let valuesA = [];
                    let valuesB = [];
                    
                    if (rowA.rows && rowA.rows.length > 0) {
                        rowA.rows.forEach(row => {
                            const rawValue = row[sortConfig.field];
                            const value = (rawValue === null || rawValue === undefined || rawValue === '') ? 0 : parseFloat(rawValue);
                            if (!isNaN(value)) {
                                sumA += value;
                                valuesA.push(value);
                            }
                        });
                    }
                    
                    if (rowB.rows && rowB.rows.length > 0) {
                        rowB.rows.forEach(row => {
                            const rawValue = row[sortConfig.field];
                            const value = (rawValue === null || rawValue === undefined || rawValue === '') ? 0 : parseFloat(rawValue);
                            if (!isNaN(value)) {
                                sumB += value;
                                valuesB.push(value);
                            }
                        });
                    }
                
                valueA = sumA;
                valueB = sumB;
                    
                    // Логируем только первые 5 сравнений для отладки
                    if (Math.random() < 0.05) {
                        console.log('Сортировка по метрике (пример):', { 
                            field: sortConfig.field, 
                            rowKeyA: a,
                            rowKeyB: b,
                            sumA, 
                            sumB,
                            rowsCountA: rowA.rows?.length || 0,
                            rowsCountB: rowB.rows?.length || 0,
                            valueA,
                            valueB,
                            sampleValuesA: valuesA.slice(0, 3),
                            sampleValuesB: valuesB.slice(0, 3)
                        });
                    }
                }
            } else {
                // Сортировка по полю строки (dimension)
                valueA = rowA.fields[sortConfig.field];
                valueB = rowB.fields[sortConfig.field];
                
                console.log('Сортировка по полю строки:', { 
                    field: sortConfig.field, 
                    valueA, 
                    valueB, 
                    type: sortConfig.type 
                });
                
                // Обработка разных типов данных
                if (sortConfig.type === 'number') {
                    valueA = parseFloat(valueA) || 0;
                    valueB = parseFloat(valueB) || 0;
                } else if (sortConfig.type === 'date') {
                    valueA = new Date(valueA) || new Date(0);
                    valueB = new Date(valueB) || new Date(0);
                } else {
                    // Текстовый тип
                    valueA = String(valueA || '').toLowerCase();
                    valueB = String(valueB || '').toLowerCase();
                }
            }
            
            let comparison = 0;
            if (valueA < valueB) comparison = -1;
            else if (valueA > valueB) comparison = 1;
            
            return sortConfig.direction === 'desc' ? -comparison : comparison;
        });
        
        // Создаем новую Map с отсортированными ключами
        const sortedRowGroups = new Map();
        sortedRowKeys.forEach(key => {
            sortedRowGroups.set(key, this.rowGroups.get(key));
        });
        
        this.rowGroups = sortedRowGroups;
        console.log('Сортировка применена, новый порядок строк:', sortedRowKeys.slice(0, 10));
        console.log('Примеры ключей rowGroups ПОСЛЕ сортировки:', Array.from(this.rowGroups.keys()).slice(0, 10));
        
        // Логируем первые 10 строк с их значениями метрики для проверки
        console.log('Первые 10 строк после сортировки с значениями:');
        sortedRowKeys.slice(0, 10).forEach(key => {
            const rowGroup = sortedRowGroups.get(key);
            if (rowGroup && rowGroup.rows) {
                let sum = 0;
                rowGroup.rows.forEach(row => {
                    const rawValue = row[sortConfig.field];
                    const value = (rawValue === null || rawValue === undefined || rawValue === '') ? 0 : parseFloat(rawValue);
                    if (!isNaN(value)) {
                        sum += value;
                    }
                });
                console.log(`  ${key}: ${sum.toFixed(2)}`);
            }
        });
        
        console.log('=== КОНЕЦ СОРТИРОВКИ ДАННЫХ ===');
    }
    
    getValue(rowKey, colKey, valueFieldName) {
        if (this.crossTable.has(rowKey) && this.crossTable.get(rowKey).has(colKey)) {
            return this.crossTable.get(rowKey).get(colKey)[valueFieldName] || 0;
        }
        return 0;
    }
    
    getRowFields(rowKey, visibleFields = null) {
        const allFields = this.rowGroups.get(rowKey)?.fields || {};
        
        // Если переданы видимые поля, возвращаем только их
        if (visibleFields) {
            const result = {};
            visibleFields.forEach(field => {
                if (allFields.hasOwnProperty(field.name)) {
                    result[field.name] = allFields[field.name];
                }
            });
            return result;
        }
        
        return allFields;
    }
    
    getColumnFields(colKey) {
        return this.columnGroups.get(colKey)?.fields || {};
    }
    
    // Применение фильтров к данным
    applyFilters(data, filters) {
        if (!filters || filters.length === 0) {
            return data;
        }
        
        return data.filter(row => {
            return filters.every(filter => filter.matches(row));
        });
    }
    
    // Расчет итоговых сумм для строки Total
    calculateTotals(config) {
        const totals = {};
        const rowKeys = this.getRowKeys();
        const columnKeys = this.getColumnKeys();
        
        // Инициализируем структуру для итогов
        config.values.forEach(valueField => {
            totals[valueField.name] = {};
            columnKeys.forEach(colKey => {
                totals[valueField.name][colKey] = 0;
            });
        });
        
        // Суммируем все значения
        rowKeys.forEach(rowKey => {
            config.values.forEach(valueField => {
                columnKeys.forEach(colKey => {
                    const value = this.getValue(rowKey, colKey, valueField.name);
                    totals[valueField.name][colKey] += value;
                });
            });
        });
        
        return totals;
    }
}

class PivotRenderer {
    constructor(containerId) {
        this.containerId = containerId;
        this.collapsedTimeFields = new Set(); // Отслеживаем свернутые временные поля
        this.collapsedSliceFields = new Set(); // Отслеживаем свернутые поля срезов
        this.collapsedRows = new Set(); // Отслеживаем свернутые строки
    }
    
    render(pivotData, config) {
        console.log('=== НАЧАЛО РЕНДЕРИНГА ===');
        console.log('Рендеринг сводной таблицы:', { config, pivotData });
        console.log('Количество строк для рендеринга:', pivotData.rowGroups.size);
        console.log('Примеры ключей строк для рендеринга:', Array.from(pivotData.rowGroups.keys()).slice(0, 10));
        
        // Получаем стек вызовов для отладки
        const stack = new Error().stack;
        console.log('Стек вызовов render:', stack.split('\n').slice(1, 4));
        
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error('Контейнер не найден:', this.containerId);
            return;
        }
        
        // Создаем HTML для сводной таблицы
        const html = this.createPivotTableHTML(pivotData, config);
        container.innerHTML = html;
        
        console.log('Сводная таблица отрендерена');
        
        // Автоматически строим график после рендеринга таблицы
        setTimeout(() => {
            if (typeof updatePivotChart === 'function') {
                console.log('Автоматическое построение графика...');
                updatePivotChart();
            }
        }, 100);
        
        console.log('=== КОНЕЦ РЕНДЕРИНГА ===');
    }
    
    createPivotTableHTML(pivotData, config) {
        let html = '<div class="pivot-table-container">';
        html += '<style>';
        html += '.collapse-btn { border: 1px solid #007bff !important; color: #007bff !important; background: white !important; }';
        html += '.collapse-btn:hover { background: #007bff !important; color: white !important; }';
        html += '.pivot-cell.expanded { background-color: #e3f2fd !important; border-left: 3px solid #2196f3 !important; }';
        html += '.pivot-cell.collapsed { background-color: #f5f5f5 !important; border-left: 3px solid #9e9e9e !important; }';
        html += '</style>';
        
        // Контейнер для графика
        html += '<div class="card mb-3">';
        html += '<div class="card-header bg-primary text-white">';
        html += '<div class="d-flex justify-content-between align-items-center">';
        html += '<h6 class="mb-0"><i class="fas fa-chart-line me-2"></i>График</h6>';
        html += '<div class="btn-group btn-group-sm" role="group">';
        html += '<button type="button" class="btn btn-light btn-sm" onclick="updatePivotChart()" title="Обновить график">';
        html += '<i class="fas fa-sync-alt"></i>';
        html += '</button>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
        html += '<div class="card-body">';
        html += '<canvas id="pivotChart" style="max-height: 400px;"></canvas>';
        html += '</div>';
        html += '</div>';
        
        html += '<div class="d-flex justify-content-between align-items-center mb-3">';
        html += '<h6 class="mb-0"><i class="fas fa-table me-2"></i>Сводная таблица (новая система)</h6>';
        html += '</div>';
        
        html += '<div class="table-responsive">';
        html += '<table class="table table-striped table-bordered pivot-table">';
        
        // Заголовки
        html += this.createHeadersHTML(pivotData, config);
        
        // Строки данных
        html += this.createRowsHTML(pivotData, config);
        
        html += '</table>';
        html += '</div>';
        html += '</div>';
        
        return html;
    }
    
    // Создание заголовка с кнопкой сортировки
    createSortableHeader(field, label, type, config, additionalClasses = '', rowspan = '', collapseIcon = '') {
        console.log(`🔍 Создаем сортируемый заголовок: ${field} (${type})`);
        console.log('Создаем сортируемый заголовок:', { field, label, type, additionalClasses });
        
        const isActive = config.sortConfig.field === field;
        const direction = isActive ? config.sortConfig.direction : 'asc';
        const sortIcon = isActive ? 
            (direction === 'asc' ? 'fa-sort-up' : 'fa-sort-down') : 
            'fa-sort';
        
        const sortButton = `
            <button class="btn btn-sm btn-link text-white p-0 ms-1" 
                    onclick="togglePivotSort('${field}', '${type}')" 
                    title="Сортировать по ${label}">
                <i class="fas ${sortIcon}"></i>
            </button>
        `;
        
        const rowspanAttr = rowspan ? `rowspan="${rowspan}"` : '';
        const headerHTML = `<th class="pivot-header ${additionalClasses}" data-field="${field}" ${rowspanAttr}>
                    ${collapseIcon}${label}${sortButton}
                </th>`;
        
        console.log('Создан заголовок с сортировкой:', headerHTML);
        return headerHTML;
    }
    
    // Определение типа поля для сортировки
    getFieldType(fieldName) {
        // Определяем тип на основе имени поля
        if (['year', 'month'].includes(fieldName.toLowerCase())) {
            return 'number';
        } else if (['date', 'datetime'].includes(fieldName.toLowerCase())) {
            return 'date';
        } else if (['revenue_first_transactions', 'revenue_repeat_transactions', 'ads_cost', 'bonus_company', 'promo_cost', 'mar_cost', 'traffic_total', 'first_transactions', 'repeat_transactions', 'transacitons_total'].includes(fieldName)) {
            return 'number';
        } else {
            return 'text';
        }
    }
    
    createHeadersHTML(pivotData, config) {
        let html = '<thead class="table-dark">';
        
        if (config.mode === 'split-columns') {
            // Режим разбивки по столбцам (как в Google Sheets)
            
            // Первый уровень заголовков - метрики
            html += '<tr>';
            
            // Определяем, какие поля использовать в строках в зависимости от исходного режима
            if (config.originalMode === 'slices') {
                // Исходный режим "срезы" - используем срезы в строках с коллапсированием и сортировкой
                const visibleSliceFields = this.getVisibleSliceFields(config);
                visibleSliceFields.forEach(rowField => {
                    const isCollapsible = this.hasChildSliceFields(config, rowField);
                    // Кнопка-триггер для всех кнопок в этом столбце
                    const collapseIcon = isCollapsible ? 
                        `<span class="collapse-icon" onclick="triggerColumnButtons('${rowField.name}', 'slice')" style="cursor: pointer; margin-right: 5px;" title="Нажать все кнопки в столбце ${rowField.label}">⚡</span>` : 
                        '<span style="margin-right: 12px;"></span>';
                    
                    // Определяем тип поля для сортировки
                    const fieldType = this.getFieldType(rowField.name);
                    const sortableHeader = this.createSortableHeader(
                        rowField.name, 
                        rowField.label, 
                        fieldType, 
                        config, 
                        'slice-header', 
                        '2',
                        collapseIcon
                    );
                    html += sortableHeader;
                });
            } else {
                // Исходный режим "временные ряды" - используем временные поля в строках с коллапсированием и сортировкой
                const visibleTimeFields = this.getVisibleTimeFields(config);
                visibleTimeFields.forEach(rowField => {
                    const isCollapsible = this.hasChildTimeFields(config, rowField);
                    // Кнопка-триггер для всех кнопок в этом столбце
                    const collapseIcon = isCollapsible ? 
                        `<span class="collapse-icon" onclick="triggerColumnButtons('${rowField.name}', 'time')" style="cursor: pointer; margin-right: 5px;" title="Нажать все кнопки в столбце ${rowField.label}">⚡</span>` : 
                        '<span style="margin-right: 12px;"></span>';
                    
                    // Определяем тип поля для сортировки
                    const fieldType = this.getFieldType(rowField.name);
                    const sortableHeader = this.createSortableHeader(
                        rowField.name, 
                        rowField.label, 
                        fieldType, 
                        config, 
                        'time-header', 
                        '2',
                        collapseIcon
                    );
                    html += sortableHeader;
                });
            }
            
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                const sortableHeader = this.createSortableHeader(
                    valueField.name, 
                    valueField.label, 
                    'number', 
                    config, 
                    'metric-header', 
                    '',
                    ''
                );
                // Заменяем обычный th на сортируемый и добавляем colspan
                const sortableHeaderWithColspan = sortableHeader.replace(
                    '<th class="pivot-header metric-header"', 
                    `<th class="pivot-header metric-header" colspan="${columnKeys.length}"`
                );
                html += sortableHeaderWithColspan;
            });
            html += '</tr>';
            
            // Второй уровень заголовков - срезы
            html += '<tr>';
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                columnKeys.forEach(colKey => {
                    const colFields = pivotData.getColumnFields(colKey);
                    const colLabel = config.columns.map(colField => colFields[colField.name]).join(' - ');
                    
                    // Создаем уникальный ключ для сортировки по конкретному столбцу метрики
                    const sortKey = `${valueField.name}_${colKey}`;
                    const sortableHeader = this.createSortableHeader(
                        sortKey, 
                        colLabel || colKey, 
                        'number', 
                        config, 
                        'slice-header', 
                        '',
                        ''
                    );
                    html += sortableHeader;
                });
            });
            html += '</tr>';
        } else if (config.mode === 'time-series') {
            // Режим временных рядов
            html += '<tr>';
            
            // Заголовки для видимых временных полей (строки) с поддержкой коллапсирования и сортировки
            const visibleTimeFields = this.getVisibleTimeFields(config);
            visibleTimeFields.forEach(rowField => {
                const isCollapsible = this.hasChildTimeFields(config, rowField);
                // Кнопка-триггер для всех кнопок в этом столбце
                const collapseIcon = isCollapsible ? 
                    `<span class="collapse-icon" onclick="triggerColumnButtons('${rowField.name}', 'time')" style="cursor: pointer; margin-right: 5px;" title="Нажать все кнопки в столбце ${rowField.label}">⚡</span>` : 
                    '<span style="margin-right: 12px;"></span>';
                
                // Определяем тип поля для сортировки
                const fieldType = this.getFieldType(rowField.name);
                const sortableHeader = this.createSortableHeader(
                    rowField.name, 
                    rowField.label, 
                    fieldType, 
                    config, 
                    'time-header', 
                    '',
                    collapseIcon
                );
                html += sortableHeader;
            });
            
            // Заголовки для метрик (значения) с сортировкой
            config.values.forEach(valueField => {
                const sortableHeader = this.createSortableHeader(
                    valueField.name, 
                    valueField.label, 
                    'number', 
                    config, 
                    'metric-header', 
                    ''
                );
                html += sortableHeader;
            });
            
            html += '</tr>';
        } else if (config.mode === 'slices') {
            // Режим срезов
            html += '<tr>';
            
            // Заголовки для видимых срезов (строки) с поддержкой коллапсирования и сортировки
            const visibleSliceFields = this.getVisibleSliceFields(config);
            visibleSliceFields.forEach(rowField => {
                const isCollapsible = this.hasChildSliceFields(config, rowField);
                // Кнопка-триггер для всех кнопок в этом столбце
                const collapseIcon = isCollapsible ? 
                    `<span class="collapse-icon" onclick="triggerColumnButtons('${rowField.name}', 'slice')" style="cursor: pointer; margin-right: 5px;" title="Нажать все кнопки в столбце ${rowField.label}">⚡</span>` : 
                    '<span style="margin-right: 12px;"></span>';
                
                // Определяем тип поля для сортировки
                const fieldType = this.getFieldType(rowField.name);
                const sortableHeader = this.createSortableHeader(
                    rowField.name, 
                    rowField.label, 
                    fieldType, 
                    config, 
                    'slice-header', 
                    '',
                    collapseIcon
                );
                html += sortableHeader;
            });
            
            // Если есть разбивка по столбцам (временные поля)
            if (config.columns.length > 0) {
                config.columns.forEach(colField => {
                    html += `<th class="pivot-header time-header" data-field="${colField.name}">${colField.label}</th>`;
                });
            }
            
            // Заголовки для метрик (значения) с сортировкой
            config.values.forEach(valueField => {
                const sortableHeader = this.createSortableHeader(
                    valueField.name, 
                    valueField.label, 
                    'number', 
                    config, 
                    'metric-header', 
                    ''
                );
                html += sortableHeader;
            });
            
            html += '</tr>';
        } else {
            // Обычный режим
            html += '<tr>';
            
            // Заголовки для строк
            config.rows.forEach(rowField => {
                html += `<th class="pivot-header">${rowField.label}</th>`;
            });
            
            // Заголовки для столбцов
            config.columns.forEach(colField => {
                html += `<th class="pivot-header">${colField.label}</th>`;
            });
            
            // Заголовки для значений
            config.values.forEach(valueField => {
                html += `<th class="pivot-header">${valueField.label}</th>`;
            });
            
            html += '</tr>';
        }
        
        html += '</thead>';
        return html;
    }
    
    createRowsHTML(pivotData, config) {
        let html = '<tbody>';
        
        // Добавляем строку Total в самом начале
        html += this.createTotalRowHTML(pivotData, config);
        
        if (config.mode === 'split-columns') {
            // Режим разбивки по столбцам с иерархической структурой
            const rowKeys = pivotData.getRowKeys();
            const columnKeys = pivotData.getColumnKeys();
            
            console.log('=== СОЗДАНИЕ СТРОК HTML (split-columns) ===');
            console.log('Количество строк для отображения:', rowKeys.length);
            console.log('Примеры ключей строк для отображения:', rowKeys.slice(0, 10));
            
            // Определяем, какие поля использовать для иерархии
            let visibleFields;
            if (config.originalMode === 'slices') {
                visibleFields = this.getVisibleSliceFields(config);
            } else {
                visibleFields = this.getVisibleTimeFields(config);
            }
            
            // Создаем правильную иерархическую структуру
            const hierarchicalRows = this.createHierarchicalStructure(rowKeys, visibleFields);
            
            // Создаем правильную иерархическую сортировку
            const sortedRows = this.createHierarchicalSorting(hierarchicalRows, pivotData.rowGroups);
            
            console.log('Отладка sortedRows (split-columns):', {
                totalRows: sortedRows.length,
                collapsedRows: this.collapsedRows || new Set(),
                collapsedRowsSize: (this.collapsedRows || new Set()).size,
                firstFewRows: sortedRows.slice(0, 5).map(([key, data]) => ({ key, isAggregated: data.isAggregated, level: data.level }))
            });
            
            sortedRows.forEach(([rowKey, rowData], index) => {
                // Проверяем видимость строки
                if (!this.isRowVisible(rowKey, this.collapsedRows || new Set())) {
                    return; // Пропускаем невидимые строки
                }
                
                // Проверяем, нужно ли добавить кнопку коллапсирования
                let shouldShowCollapseButton = false;
                
                // Проверяем, есть ли кнопки в данных строки
                if (rowData.collapseButtons && rowData.collapseButtons.length > 0) {
                    shouldShowCollapseButton = true;
                    console.log(`✅ Найдены кнопки для строки ${rowKey}: ${rowData.collapseButtons.length} кнопок`);
                }
                
                const rowFields = rowData.fields;
                
                html += '<tr>';
                
                // Определяем, какие поля использовать в строках в зависимости от исходного режима
                if (config.originalMode === 'slices') {
                    // Исходный режим "срезы" - используем срезы в строках
                    visibleFields.forEach((rowField, index) => {
                        let cellContent = rowFields[rowField.name] || '';
                        
                        let cellClass = 'pivot-cell';
                        
                        // Кнопка для агрегированных строк (свернутое состояние)
                        if (rowData.isAggregated && index === rowData.level) {
                            const collapseKey = rowKey;
                            const collapseIcon = '+';
                            
                            cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                            
                            cellClass += ' collapsed';
                        }
                        // Кнопки для первой дочерней строки (развернутое состояние)
                        else if (shouldShowCollapseButton && rowData.collapseButtons) {
                            const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
                            
                            if (buttonForLevel) {
                                const collapseIcon = buttonForLevel.collapseIcon;
                                const collapseKey = buttonForLevel.collapseKey;
                                
                                cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                                
                                cellClass += ' expanded';
                            }
                        }
                            
                        html += `<td class="${cellClass}">${cellContent}</td>`;
                    });
                } else {
                    // Исходный режим "временные ряды" - используем временные поля в строках
                    visibleFields.forEach((rowField, index) => {
                        let cellContent = rowFields[rowField.name] || '';
                        
                        let cellClass = 'pivot-cell';
                        
                        // Кнопка для агрегированных строк (свернутое состояние)
                        if (rowData.isAggregated && index === rowData.level) {
                            const collapseKey = rowKey;
                            const collapseIcon = '+';
                            
                            cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                            
                            cellClass += ' collapsed';
                        }
                        // Кнопки для первой дочерней строки (развернутое состояние)
                        else if (shouldShowCollapseButton && rowData.collapseButtons) {
                            const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
                            
                            if (buttonForLevel) {
                                const collapseIcon = buttonForLevel.collapseIcon;
                                const collapseKey = buttonForLevel.collapseKey;
                                
                                cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                                
                                cellClass += ' expanded';
                            }
                        }
                            
                        html += `<td class="${cellClass}">${cellContent}</td>`;
                    });
                }
                
                // Значения для каждой метрики и каждого столбца (среза)
                config.values.forEach(valueField => {
                    columnKeys.forEach(colKey => {
                        let value = 0;
                        
                        if (rowData.isAggregated) {
                            // Для агрегированных строк суммируем значения из ВСЕХ дочерних элементов (рекурсивно)
                            let childCount = 0;
                            
                            // В режиме split-columns ищем все строки, которые начинаются с текущего ключа
                            // Берем ВСЕ дочерние элементы, не только непосредственных детей
                            const allRowKeys = pivotData.getRowKeys();
                            
                            allRowKeys.forEach(childKey => {
                                // Проверяем, что это дочерний элемент (любого уровня вложенности)
                                if (childKey.startsWith(rowKey + '|')) {
                                    const childValue = pivotData.getValue(childKey, colKey, valueField.name);
                                    value += childValue;
                                    childCount++;
                                    
                                    // Отладочный лог для первых нескольких случаев
                                    if (Math.random() < 0.01) {
                                        console.log(`Агрегация для ${rowKey}/${colKey}: дочерний элемент ${childKey} = ${childValue}, общая сумма = ${value}`);
                                    }
                                }
                            });
                            
                            // Отладочный лог для агрегированных строк
                            if (Math.random() < 0.01) {
                                console.log(`Агрегированная строка ${rowKey}, столбец ${colKey}: найдено ${childCount} дочерних элементов, итоговая сумма = ${value}`);
                            }
                        } else {
                            // Для обычных строк берем значение из crossTable
                            value = pivotData.getValue(rowKey, colKey, valueField.name);
                        }
                        
                        html += `<td class="pivot-cell" style="text-align: right;">${this.formatValue(value)}</td>`;
                    });
                });
                
                html += '</tr>';
            });
        } else if (config.mode === 'time-series') {
            // Режим временных рядов
            const rowKeys = pivotData.getRowKeys();
            const visibleTimeFields = this.getVisibleTimeFields(config);
            
            console.log('=== СОЗДАНИЕ СТРОК HTML (time-series) ===');
            console.log('Количество строк для отображения:', rowKeys.length);
            console.log('Примеры ключей строк для отображения:', rowKeys.slice(0, 10));
            
            // Создаем правильную иерархическую структуру
            const hierarchicalRows = this.createHierarchicalStructure(rowKeys, visibleTimeFields);
            
            // Создаем правильную иерархическую сортировку
            const sortedRows = this.createHierarchicalSorting(hierarchicalRows, pivotData.rowGroups);
            
            console.log('Отладка sortedRows:', {
                totalRows: sortedRows.length,
                collapsedRows: this.collapsedRows || new Set(),
                collapsedRowsSize: (this.collapsedRows || new Set()).size,
                firstFewRows: sortedRows.slice(0, 5).map(([key, data]) => ({ key, isAggregated: data.isAggregated, level: data.level }))
            });
            
            sortedRows.forEach(([rowKey, rowData], index) => {
                // Проверяем видимость строки
                if (!this.isRowVisible(rowKey, this.collapsedRows || new Set())) {
                    return; // Пропускаем невидимые строки
                }
                
                // Проверяем, нужно ли добавить кнопку коллапсирования
                let shouldShowCollapseButton = false;
                let collapseKey = '';
                let isCollapsed = false;
                
                // Проверяем, есть ли кнопки в данных строки
                if (rowData.collapseButtons && rowData.collapseButtons.length > 0) {
                    shouldShowCollapseButton = true;
                    console.log(`✅ Найдены кнопки для строки ${rowKey}: ${rowData.collapseButtons.length} кнопок`);
                }
                
                const rowFields = rowData.fields;
                
                html += '<tr>';
                
                    // Временные поля (строки)
                visibleTimeFields.forEach((rowField, index) => {
                        let cellContent = rowFields[rowField.name] || '';
                        
                        let cellClass = 'pivot-cell';
                        
                        // Кнопка для агрегированных строк (свернутое состояние)
                        if (rowData.isAggregated && index === rowData.level) {
                            const collapseKey = rowKey;
                            const isCollapsed = (this.collapsedRows || new Set()).has(collapseKey);
                            const collapseIcon = '+'; // Всегда плюс для агрегированных строк
                            
                            cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                            
                            console.log(`Добавлена кнопка ${collapseIcon} для строки ${rowKey} (агрегированная строка)`);
                            
                            // Добавляем класс для выделения свернутого состояния
                            cellClass += ' collapsed';
                        }
                        // Кнопки для первой дочерней строки (развернутое состояние)
                        else if (shouldShowCollapseButton && rowData.collapseButtons) {
                            // Ищем кнопку для текущего уровня
                            const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
                            
                            if (buttonForLevel) {
                                const collapseIcon = buttonForLevel.collapseIcon;
                                const collapseKey = buttonForLevel.collapseKey;
                                
                                cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                                
                                console.log(`Добавлена кнопка ${collapseIcon} для строки ${rowKey} в столбце уровня ${index} (коллапс до ${collapseKey})`);
                                
                                // Добавляем класс для выделения развернутого состояния
                                cellClass += ' expanded';
                            }
                        }
                            
                        html += `<td class="${cellClass}">${cellContent}</td>`;
                    });
                
                // Значения метрик
                config.values.forEach(valueField => {
                    let aggregatedValue = 0;
                    
                    if (rowData.isAggregated) {
                        // Для агрегированных строк суммируем все дочерние значения
                        // Но только те, которые являются непосредственными дочерними элементами
                        rowKeys.forEach(childKey => {
                            if (childKey.startsWith(rowKey + '|')) {
                                // Проверяем, что это непосредственный дочерний элемент
                                const childKeyParts = childKey.split('|');
                                const parentKeyParts = rowKey.split('|');
                                
                                // Если дочерний ключ на один уровень глубже родительского
                                if (childKeyParts.length === parentKeyParts.length + 1) {
                                    const childGroup = pivotData.rowGroups.get(childKey);
                                    if (childGroup && childGroup.rows) {
                                        childGroup.rows.forEach(row => {
                            const value = parseFloat(row[valueField.name]) || 0;
                            aggregatedValue += value;
                                        });
                                    }
                                }
                            }
                        });
                    } else {
                        // Для детализированных строк используем значения из группы
                        const rowGroup = pivotData.rowGroups.get(rowKey);
                        if (rowGroup && rowGroup.rows) {
                            rowGroup.rows.forEach(row => {
                                const value = parseFloat(row[valueField.name]) || 0;
                                aggregatedValue += value;
                            });
                        }
                    }
                    
                    html += `<td class="pivot-cell" style="text-align: right;">${this.formatValue(aggregatedValue)}</td>`;
                });
                
                html += '</tr>';
            });
        } else if (config.mode === 'slices') {
            // Режим срезов с иерархической структурой
            const rowKeys = pivotData.getRowKeys();
            const visibleSliceFields = this.getVisibleSliceFields(config);
            
            console.log('=== СОЗДАНИЕ СТРОК HTML (slices) ===');
            console.log('Количество строк для отображения:', rowKeys.length);
            console.log('Примеры ключей строк для отображения:', rowKeys.slice(0, 10));
            
            // Создаем правильную иерархическую структуру
            const hierarchicalRows = this.createHierarchicalStructure(rowKeys, visibleSliceFields);
            
            // Создаем правильную иерархическую сортировку
            const sortedRows = this.createHierarchicalSorting(hierarchicalRows, pivotData.rowGroups);
            
            console.log('Отладка sortedRows (slices):', {
                totalRows: sortedRows.length,
                collapsedRows: this.collapsedRows || new Set(),
                collapsedRowsSize: (this.collapsedRows || new Set()).size,
                firstFewRows: sortedRows.slice(0, 5).map(([key, data]) => ({ key, isAggregated: data.isAggregated, level: data.level }))
            });
            
            sortedRows.forEach(([rowKey, rowData], index) => {
                // Проверяем видимость строки
                if (!this.isRowVisible(rowKey, this.collapsedRows || new Set())) {
                    return; // Пропускаем невидимые строки
                }
                
                // Проверяем, нужно ли добавить кнопку коллапсирования
                let shouldShowCollapseButton = false;
                
                // Проверяем, есть ли кнопки в данных строки
                if (rowData.collapseButtons && rowData.collapseButtons.length > 0) {
                    shouldShowCollapseButton = true;
                    console.log(`✅ Найдены кнопки для строки ${rowKey}: ${rowData.collapseButtons.length} кнопок`);
                }
                
                const rowFields = rowData.fields;
                
                html += '<tr>';
                
                // Поля срезов (строки)
                visibleSliceFields.forEach((rowField, index) => {
                    let cellContent = rowFields[rowField.name] || '';
                    
                    let cellClass = 'pivot-cell';
                    
                    // Кнопка для агрегированных строк (свернутое состояние)
                    if (rowData.isAggregated && index === rowData.level) {
                        const collapseKey = rowKey;
                        const isCollapsed = (this.collapsedRows || new Set()).has(collapseKey);
                        const collapseIcon = '+'; // Всегда плюс для агрегированных строк
                        
                        cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                        
                        console.log(`Добавлена кнопка ${collapseIcon} для строки ${rowKey} (агрегированная строка срезов)`);
                        
                        // Добавляем класс для выделения свернутого состояния
                        cellClass += ' collapsed';
                    }
                    // Кнопки для первой дочерней строки (развернутое состояние)
                    else if (shouldShowCollapseButton && rowData.collapseButtons) {
                        // Ищем кнопку для текущего уровня
                        const buttonForLevel = rowData.collapseButtons.find(btn => btn.level === index);
                        
                        if (buttonForLevel) {
                            const collapseIcon = buttonForLevel.collapseIcon;
                            const collapseKey = buttonForLevel.collapseKey;
                            
                            cellContent = `<button class="btn btn-sm btn-outline-primary collapse-btn" onclick="toggleRowCollapse('${collapseKey}')" style="margin-right: 8px; padding: 2px 6px; font-size: 12px; border-radius: 3px; min-width: 20px;">${collapseIcon}</button>${cellContent}`;
                            
                            console.log(`Добавлена кнопка ${collapseIcon} для строки ${rowKey} в столбце уровня ${index} (коллапс до ${collapseKey})`);
                            
                            // Добавляем класс для выделения развернутого состояния
                            cellClass += ' expanded';
                        }
                    }
                        
                    html += `<td class="${cellClass}">${cellContent}</td>`;
                });
                
                // Значения метрик
                config.values.forEach(valueField => {
                    let aggregatedValue = 0;
                    
                    if (rowData.isAggregated) {
                        // Для агрегированных строк суммируем все дочерние значения
                        // Но только те, которые являются непосредственными дочерними элементами
                        rowKeys.forEach(childKey => {
                            if (childKey.startsWith(rowKey + '|')) {
                                // Проверяем, что это непосредственный дочерний элемент
                                const childKeyParts = childKey.split('|');
                                const parentKeyParts = rowKey.split('|');
                                
                                // Если дочерний ключ на один уровень глубже родительского
                                if (childKeyParts.length === parentKeyParts.length + 1) {
                                    const childGroup = pivotData.rowGroups.get(childKey);
                                    if (childGroup && childGroup.rows) {
                                        childGroup.rows.forEach(row => {
                                            const value = parseFloat(row[valueField.name]) || 0;
                                            aggregatedValue += value;
                                        });
                                    }
                                }
                            }
                        });
                    } else {
                        // Для обычных строк суммируем значения из группы
                        const rowGroup = pivotData.rowGroups.get(rowKey);
                    if (rowGroup && rowGroup.rows) {
                        rowGroup.rows.forEach(row => {
                            const value = parseFloat(row[valueField.name]) || 0;
                            aggregatedValue += value;
                        });
                        }
                    }
                    
                    html += `<td class="pivot-cell" style="text-align: right;">${this.formatValue(aggregatedValue)}</td>`;
                });
                
                html += '</tr>';
            });
        } else {
            // Обычный режим
            const rowKeys = pivotData.getRowKeys();
            
            rowKeys.forEach(rowKey => {
                html += '<tr>';
                
                // Значения строк
                const rowFields = pivotData.getRowFields(rowKey);
                config.rows.forEach(rowField => {
                    html += `<td class="pivot-cell">${rowFields[rowField.name] || ''}</td>`;
                });
                
                // Значения столбцов
                const columnKeys = pivotData.getColumnKeys();
                columnKeys.forEach(colKey => {
                    const colFields = pivotData.getColumnFields(colKey);
                    config.columns.forEach(colField => {
                        html += `<td class="pivot-cell">${colFields[colField.name] || ''}</td>`;
                    });
                });
                
                // Значения метрик - агрегируем по всем строкам в группе
                config.values.forEach(valueField => {
                    const rowGroup = pivotData.rowGroups.get(rowKey);
                    let aggregatedValue = 0;
                    
                    // Суммируем все значения метрики в группе строк
                    if (rowGroup && rowGroup.rows) {
                        rowGroup.rows.forEach(row => {
                            const value = parseFloat(row[valueField.name]) || 0;
                            aggregatedValue += value;
                        });
                    }
                    
                    html += `<td class="pivot-cell" style="text-align: right;">${this.formatValue(aggregatedValue)}</td>`;
                });
                
                html += '</tr>';
            });
        }
        
        html += '</tbody>';
        return html;
    }
    
    formatValue(value) {
        if (typeof value === 'number') {
            return new Intl.NumberFormat('ru-RU', {
                minimumFractionDigits: 0,
                maximumFractionDigits: 2
            }).format(value);
        }
        return value || '';
    }
    
    // Создание HTML для строки Total
    createTotalRowHTML(pivotData, config) {
        let html = '<tr class="pivot-total-row">';
        
        if (config.mode === 'split-columns') {
            // Режим разбивки по столбцам
            const columnKeys = pivotData.getColumnKeys();
            const totals = pivotData.calculateTotals(config);
            
            // Определяем, какие поля использовать в строках в зависимости от исходного режима
            if (config.originalMode === 'slices') {
                // Исходный режим "срезы" - используем срезы в строках
                const visibleSliceFields = this.getVisibleSliceFields(config);
                visibleSliceFields.forEach((field, index) => {
                    const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                    html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
                });
            } else {
                // Исходный режим "временные ряды" - используем временные поля в строках
                const visibleTimeFields = this.getVisibleTimeFields(config);
                visibleTimeFields.forEach((field, index) => {
                    const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                    html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
                });
            }
            
            // Итоговые значения для каждой метрики и каждого столбца (среза)
            config.values.forEach(valueField => {
                columnKeys.forEach(colKey => {
                    const totalValue = totals[valueField.name][colKey] || 0;
                    html += `<td class="pivot-cell pivot-total-cell" style="text-align: right; font-weight: bold;">${this.formatValue(totalValue)}</td>`;
                });
            });
            
        } else if (config.mode === 'time-series') {
            // Режим временных рядов
            const visibleTimeFields = this.getVisibleTimeFields(config);
            const totals = pivotData.calculateTotals(config);
            
            // Ячейки для временных полей с "Total" в первой
            visibleTimeFields.forEach((field, index) => {
                const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
            });
            
            // Итоговые значения для метрик (суммируем по всем столбцам)
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                let totalValue = 0;
                columnKeys.forEach(colKey => {
                    totalValue += totals[valueField.name][colKey] || 0;
                });
                html += `<td class="pivot-cell pivot-total-cell" style="text-align: right; font-weight: bold;">${this.formatValue(totalValue)}</td>`;
            });
        } else if (config.mode === 'slices') {
            // Режим срезов
            const totals = pivotData.calculateTotals(config);
            const visibleSliceFields = this.getVisibleSliceFields(config);
            
            // Ячейки для видимых срезов с "Total" в первой
            visibleSliceFields.forEach((field, index) => {
                const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
            });
            
            // Если есть разбивка по столбцам (временные поля)
            if (config.columns.length > 0) {
                config.columns.forEach((field, index) => {
                    const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                    html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
                });
            }
            
            // Итоговые значения для метрик (суммируем по всем столбцам)
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                let totalValue = 0;
                columnKeys.forEach(colKey => {
                    totalValue += totals[valueField.name][colKey] || 0;
                });
                html += `<td class="pivot-cell pivot-total-cell" style="text-align: right; font-weight: bold;">${this.formatValue(totalValue)}</td>`;
            });
        } else {
            // Обычный режим
            const visibleTimeFields = this.getVisibleTimeFields(config);
            const totals = pivotData.calculateTotals(config);
            
            // Ячейки для временных полей с "Total" в первой
            visibleTimeFields.forEach((field, index) => {
                const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
            });
            
            // Итоговые значения для метрик
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                let totalValue = 0;
                columnKeys.forEach(colKey => {
                    totalValue += totals[valueField.name][colKey] || 0;
                });
                html += `<td class="pivot-cell pivot-total-cell" style="text-align: right; font-weight: bold;">${this.formatValue(totalValue)}</td>`;
            });
        }
        
        html += '</tr>';
        return html;
    }
    
    // Методы для работы с коллапсированием временных полей
    hasChildTimeFields(config, field) {
        // Проверяем, есть ли дочерние временные поля с более высоким уровнем
        return config.rows.some(otherField => 
            otherField.type === 'time' && otherField.level > field.level
        );
    }
    
    hasChildSliceFields(config, field) {
        // Проверяем, есть ли дочерние поля срезов с более высоким уровнем
        return config.rows.some(otherField => 
            otherField.type === 'slice' && otherField.level > field.level
        );
    }
    
    isTimeFieldCollapsed(fieldName) {
        // Поле считается свернутым если НЕ в collapsedTimeFields
        return !this.collapsedTimeFields.has(fieldName);
    }
    
    isSliceFieldCollapsed(fieldName) {
        return this.collapsedSliceFields.has(fieldName);
    }
    
    // Методы toggleTimeFieldCollapse и toggleSliceFieldCollapse удалены
    // Теперь используется triggerColumnButtons для триггера всех кнопок в столбце
    
    getVisibleTimeFields(config) {
        // В режиме срезов нет временных полей в строках
        if (config.mode === 'slices') {
            return [];
        }
        
        // Возвращаем только видимые временные поля (не свернутые)
        return config.rows.filter(field => {
            if (field.type !== 'time') return true;
            
            // Проверяем, не свернут ли родительский элемент
            const parentField = config.rows.find(parent => 
                parent.type === 'time' && parent.level < field.level && 
                this.collapsedTimeFields.has(parent.name)
            );
            
            return !parentField;
        });
    }
    
    hasChildrenForField(rowKeys, fieldName, fieldValue, fieldIndex, visibleTimeFields) {
        // Проверяем, есть ли дочерние элементы для конкретного поля
        // Например, для Year=2024 ищем строки с разными Halfyear (H1, H2)
        
        // Если это последний уровень (например, month), то дочерних элементов нет
        if (fieldIndex >= visibleTimeFields.length - 1) {
            return false;
        }
        
        const currentFieldParts = fieldValue.split('|');
        const uniqueValues = new Set();
        
        rowKeys.forEach(key => {
            const keyParts = key.split('|');
            
            // Проверяем, что у нас есть данные для этого уровня
            if (keyParts.length <= fieldIndex) return;
            
            // Проверяем, что все предыдущие поля совпадают
            let isChild = true;
            for (let i = 0; i < fieldIndex; i++) {
                if (keyParts[i] !== currentFieldParts[i]) {
                    isChild = false;
                    break;
                }
            }
            
            // Если это дочерняя строка, добавляем значение текущего поля
            if (isChild && keyParts.length > fieldIndex) {
                uniqueValues.add(keyParts[fieldIndex]);
            }
        });
        
        const hasChildren = uniqueValues.size > 0;
        console.log(`hasChildrenForField: ${fieldName}=${fieldValue} -> ${hasChildren} (${uniqueValues.size} уникальных значений: ${Array.from(uniqueValues).join(', ')})`);
        return hasChildren;
    }
    
    // Создаем правильную структуру строк как в Google Sheets
    createHierarchicalStructure(rowKeys, visibleTimeFields) {
        const allRows = new Map();
        
        // Собираем все уникальные комбинации для каждого уровня
        const levelCombinations = new Map();
        
        rowKeys.forEach(key => {
            const keyParts = key.split('|');
            
            // Создаем комбинации для всех уровней
            for (let level = 0; level < keyParts.length; level++) {
                const levelKey = keyParts.slice(0, level + 1).join('|');
                
                if (!levelCombinations.has(level)) {
                    levelCombinations.set(level, new Set());
                }
                levelCombinations.get(level).add(levelKey);
            }
        });
        
        // Создаем строки для каждого уровня
        levelCombinations.forEach((combinations, level) => {
            combinations.forEach(combination => {
                const keyParts = combination.split('|');
                
                // Создаем поля для строки
                const fields = {};
                visibleTimeFields.forEach((field, index) => {
                    if (index < keyParts.length) {
                        fields[field.name] = keyParts[index];
                    } else {
                        fields[field.name] = '';
                    }
                });
                
                // Определяем, является ли это агрегированной строкой
                const isAggregated = level < visibleTimeFields.length - 1;
                
                allRows.set(combination, {
                    key: combination,
                    fields: fields,
                    level: level,
                    isAggregated: isAggregated,
                    rows: [] // Будет заполнено позже
                });
            });
        });
        
        console.log('Создана иерархическая структура:', {
            totalRows: allRows.size,
            levels: levelCombinations.size,
            collapsedRows: this.collapsedRows || new Set(),
            collapsedRowsSize: (this.collapsedRows || new Set()).size
        });
        
        return allRows;
    }
    
    // Создаем правильную иерархическую сортировку - дочерние элементы идут сразу после родителя
    createHierarchicalSorting(hierarchicalRows, rowGroups) {
        const sortedRows = [];
        const processedKeys = new Set();
        
        console.log('=== НАЧАЛО ИЕРАРХИЧЕСКОЙ СОРТИРОВКИ ===');
        console.log('Количество hierarchicalRows:', hierarchicalRows.size);
        console.log('Примеры hierarchicalRows:', Array.from(hierarchicalRows.keys()).slice(0, 5));
        
        // Получаем все строки уровня 0 (корневые) в том же порядке, что и в rowGroups после сортировки
        const rootRows = Array.from(hierarchicalRows.entries())
            .filter(([key, rowData]) => rowData.level === 0);
        
        // Сортируем корневые строки в том же порядке, что и rowGroups
        // Получаем порядок из rowGroups (уже отсортированных)
        const rowGroupsOrder = Array.from(rowGroups.keys());
        const sortedRootRows = rootRows.sort(([keyA, rowA], [keyB, rowB]) => {
            // Находим позиции в отсортированном rowGroups
            // Ищем ТОЧНОЕ совпадение для агрегированных строк
            const indexA = rowGroupsOrder.indexOf(keyA);
            const indexB = rowGroupsOrder.indexOf(keyB);
            
            console.log(`Сравниваем корневые строки: ${keyA} (позиция ${indexA}) vs ${keyB} (позиция ${indexB})`);
            
            // Если не найдено точное совпадение, ищем первую дочернюю строку
            const finalIndexA = indexA !== -1 ? indexA : rowGroupsOrder.findIndex(rowKey => rowKey.startsWith(keyA + '|'));
            const finalIndexB = indexB !== -1 ? indexB : rowGroupsOrder.findIndex(rowKey => rowKey.startsWith(keyB + '|'));
            
            console.log(`Финальные позиции: ${keyA} (${finalIndexA}) vs ${keyB} (${finalIndexB})`);
            
            return finalIndexA - finalIndexB;
        });
        
        console.log('Отсортированные корневые строки:', sortedRootRows.map(([key, data]) => key));
        
        // Рекурсивно добавляем строки в правильном порядке
        const addRowsRecursively = (currentKey, level) => {
            // Проверяем, свернута ли текущая строка
            // Строка свернута если она НЕ в collapsedRows
            const isCurrentCollapsed = !(this.collapsedRows || new Set()).has(currentKey);
            
            console.log(`Отладка addRowsRecursively: currentKey=${currentKey}, isCurrentCollapsed=${isCurrentCollapsed}, level=${level}`);
            
            // Добавляем текущую строку только если она свернута
            if (hierarchicalRows.has(currentKey) && !processedKeys.has(currentKey)) {
                const rowData = hierarchicalRows.get(currentKey);
                
                // Если строка свернута - показываем её
                if (isCurrentCollapsed) {
                    sortedRows.push([currentKey, rowData]);
                    processedKeys.add(currentKey);
                    console.log(`Добавлена свернутая строка: ${currentKey}`);
                } else {
                    console.log(`Пропущена раскрытая строка: ${currentKey}`);
                }
                // Отмечаем как обработанную
                processedKeys.add(currentKey);
            }
            
            // Находим все дочерние строки этого уровня
            const childRows = Array.from(hierarchicalRows.entries())
                .filter(([key, rowData]) => {
                    if (processedKeys.has(key)) return false;
                    
                    // Проверяем, что это дочерний элемент
                    if (!key.startsWith(currentKey + '|')) return false;
                    
                    // Проверяем, что это непосредственный дочерний элемент
                    const keyParts = key.split('|');
                    const currentKeyParts = currentKey.split('|');
                    
                    return keyParts.length === currentKeyParts.length + 1;
                })
                .sort(([keyA, rowA], [keyB, rowB]) => {
                    // Используем порядок из rowGroups (уже отсортированных)
                    const indexA = rowGroupsOrder.findIndex(rowKey => rowKey === keyA);
                    const indexB = rowGroupsOrder.findIndex(rowKey => rowKey === keyB);
                    
                    console.log(`Сравниваем дочерние элементы: ${keyA} (позиция ${indexA}) vs ${keyB} (позиция ${indexB})`);
                    
                    return indexA - indexB;
                });
            
            // Добавляем дочерние элементы с кнопками для всех уровней родителей
            console.log(`Найдено дочерних элементов для ${currentKey}: ${childRows.length}`);
            childRows.forEach(([childKey, childData], index) => {
                // На первой дочерней строке добавляем кнопки - для всех раскрытых уровней родителей
                if (index === 0) {
                    // Создаем массив кнопок для всех уровней (если еще не создан)
                    if (!childData.collapseButtons) {
                        childData.collapseButtons = [];
                    }
                    
                    // Добавляем кнопку для текущего уровня (самого глубокого раскрытого)
                    childData.collapseButtons.push({
                        collapseKey: currentKey,
                        collapseIcon: '-',
                        level: level
                    });
                    
                    console.log(`Добавлена кнопка - для первой дочерней строки: ${childKey} на уровне ${level} (коллапс до ${currentKey})`);
                    
                    // ВАЖНО: Добавляем кнопки для всех родительских уровней
                    const childKeyParts = childKey.split('|');
                    for (let parentLevel = 0; parentLevel < level; parentLevel++) {
                        const parentKey = childKeyParts.slice(0, parentLevel + 1).join('|');
                        
                        // Проверяем, что родитель существует и раскрыт
                        if (hierarchicalRows.has(parentKey) && (this.collapsedRows || new Set()).has(parentKey)) {
                            // Добавляем кнопку для этого родительского уровня
                            childData.collapseButtons.push({
                                collapseKey: parentKey,
                                collapseIcon: '-',
                                level: parentLevel
                            });
                            
                            console.log(`Добавлена кнопка - для родительского уровня ${parentLevel}: ${parentKey} (коллапс до ${parentKey})`);
                        }
                    }
                }
                
                console.log(`Рекурсивно добавляем дочерний элемент: ${childKey}`);
                addRowsRecursively(childKey, level + 1);
            });
        };
        
        // Начинаем с корневых строк (в порядке сортировки)
        sortedRootRows.forEach(([rootKey, rootData]) => {
            addRowsRecursively(rootKey, 0);
        });
        
        console.log('Создана иерархическая сортировка:', {
            totalRows: sortedRows.length,
            processedKeys: processedKeys.size
        });
        console.log('Порядок строк в иерархической сортировке:', sortedRows.map(([key, data]) => key));
        console.log('=== КОНЕЦ ИЕРАРХИЧЕСКОЙ СОРТИРОВКИ ===');
        
        return sortedRows;
    }
    
    isRowVisible(rowKey, collapsedRows) {
        const rowFields = rowKey.split('|');
        
        // Проверяем каждый уровень родительских элементов
        for (let level = 0; level < rowFields.length - 1; level++) {
            const parentKey = rowFields.slice(0, level + 1).join('|');
            // Если родитель НЕ в collapsedRows, значит он свернут - скрываем дочерние элементы
            if (!collapsedRows.has(parentKey)) {
                console.log(`Строка ${rowKey} скрыта из-за свернутого родителя ${parentKey}`);
                return false; // Родительский элемент свернут
            }
        }
        
        return true; // Все родительские элементы развернуты
    }
    
    toggleRowCollapse(rowKey) {
        if (this.collapsedRows.has(rowKey)) {
            this.collapsedRows.delete(rowKey);
        } else {
            this.collapsedRows.add(rowKey);
        }
        
        // Перерендериваем таблицу
        if (window.currentPivotData && window.currentPivotConfig) {
            this.render(window.currentPivotData, window.currentPivotConfig);
        }
    }
    
    getVisibleSliceFields(config) {
        // В режиме срезов или split-columns из срезов возвращаем только видимые поля срезов (не свернутые)
        if (config.mode === 'slices' || (config.mode === 'split-columns' && config.originalMode === 'slices')) {
            return config.rows.filter(field => {
                if (field.type !== 'slice') return true;
                
                // Проверяем, не свернут ли родительский элемент
                const parentField = config.rows.find(parent => 
                    parent.type === 'slice' && parent.level < field.level && 
                    this.collapsedSliceFields.has(parent.name)
                );
                
                return !parentField;
            });
        }
        
        return [];
    }
}

// Функции для работы с новой системой
function createFiltersFromMapping(mappingData) {
    const filters = [];
    
    if (!mappingData || !mappingData.columns) {
        return filters;
    }
    
    mappingData.columns.forEach(col => {
        // Создаем фильтры для временных полей и срезов (dimensions)
        if (col.role === 'dimension' && col.include) {
            let fieldType = 'text';
            
            // Определяем тип поля для фильтра
            if (col.type === 'numeric') {
                fieldType = 'number';
            } else if (col.type === 'date') {
                fieldType = 'date';
            } else if (col.type === 'text') {
                // Для текстовых полей (включая временные поля с текстовыми значениями)
                fieldType = 'text';
            }
            
            const filter = new PivotFilter(col.name, fieldType, col.name);
            filters.push(filter);
            console.log('Создан фильтр:', { name: col.name, type: fieldType, time_series: col.time_series });
        }
    });
    
    return filters;
}

function createPivotConfigFromMapping(mappingData, mode = 'normal', splitBySlice = '', originalMode = '') {
    console.log('Создание конфигурации сводной таблицы из маппинга:', { mappingData, mode, splitBySlice, originalMode });
    
    const config = new PivotConfig();
    config.setMode(mode);
    config.setOriginalMode(originalMode);
    
    if (!mappingData || !mappingData.columns) {
        console.error('Нет данных маппинга');
        return config;
    }
    
    // Определяем поля для строк, столбцов и значений
    const timeFields = [];
    const sliceFields = [];
    const metricFields = [];
    
    console.log('Обрабатываем колонки маппинга:', mappingData.columns);
    
    mappingData.columns.forEach(col => {
        console.log('Обрабатываем колонку:', col);
        if (col.role === 'metric') {
            // Добавляем метрики из выбранных пользователем
            if (mappingData.selectedMetrics && mappingData.selectedMetrics.includes(col.name)) {
                metricFields.push(new PivotField(col.name, col.name, 'metric'));
                console.log('Добавлена выбранная метрика:', col.name);
            } else if (!mappingData.selectedMetrics && metricFields.length === 0) {
                // Fallback: если нет выбранных метрик, добавляем первую
                metricFields.push(new PivotField(col.name, col.name, 'metric'));
                console.log('Добавлена первая метрика (fallback):', col.name);
            }
        } else if (col.time_series && col.time_series !== '') {
            timeFields.push(new PivotField(col.name, col.name, 'time', col.nesting_level || 0));
            console.log('Добавлено временное поле:', col.name);
        } else if (col.role === 'dimension' && col.include) {
            // Добавляем измерения как срезы
            sliceFields.push(new PivotField(col.name, col.name, 'slice', col.nesting_level || 0));
            console.log('Добавлено поле среза:', col.name);
        }
    });
    
    console.log('Результат обработки полей:', {
        timeFields: timeFields.length,
        sliceFields: sliceFields.length,
        metricFields: metricFields.length,
        timeFieldsNames: timeFields.map(f => f.name),
        sliceFieldsNames: sliceFields.map(f => f.name),
        metricFieldsNames: metricFields.map(f => f.name)
    });
    
    console.log('Поля срезов до сортировки:', sliceFields.map(f => ({ name: f.name, level: f.level })));
    
    // Сортируем временные поля по уровню
    timeFields.sort((a, b) => a.level - b.level);
    
    // Сортируем поля срезов по уровню
    sliceFields.sort((a, b) => a.level - b.level);
    
    console.log('Поля срезов после сортировки:', sliceFields.map(f => ({ name: f.name, level: f.level })));
    
    // Конфигурируем в зависимости от режима
    if (mode === 'split-columns' && splitBySlice) {
        // Режим разбивки по столбцам - определяем по исходному режиму
        if (originalMode === 'slices') {
            // Исходный режим "срезы" - срезы в строках, временной ряд в столбцах
            const splitField = timeFields.find(field => field.name === splitBySlice);
            console.log('DEBUG: splitField найден:', splitField);
            console.log('DEBUG: sliceFields:', sliceFields.map(f => f.name));
            console.log('DEBUG: timeFields:', timeFields.map(f => f.name));
            if (splitField) {
                config.setRows(sliceFields);
                config.setColumns([splitField]);
                config.setValues(metricFields);
                console.log('Режим split-columns (из срезов):', { 
                    sliceFields: sliceFields.length, 
                    sliceFieldsNames: sliceFields.map(f => f.name),
                    columns: 1, 
                    values: metricFields.length 
                });
            } else {
                console.error('ERROR: splitField не найден для', splitBySlice);
            }
        } else {
            // Исходный режим "временные ряды" - временные ряды в строках, срез в столбцах
            config.setRows(timeFields);
            config.setColumns([new PivotField(splitBySlice, splitBySlice, 'slice')]);
            config.setValues(metricFields);
            console.log('Режим split-columns (из временных рядов):', { timeFields: timeFields.length, columns: 1, values: metricFields.length });
        }
    } else if (mode === 'time-series') {
        // Режим временных рядов - временные поля в строках
        config.setRows(timeFields);
        config.setColumns([]);
        config.setValues(metricFields);
        console.log('Режим time-series:', { timeFields: timeFields.length, columns: 0, values: metricFields.length });
    } else if (mode === 'slices') {
        // Режим срезов - срезы в строках, временные поля для разбивки (если есть)
        if (splitBySlice) {
            // Разбивка по временному ряду
            const splitField = timeFields.find(field => field.name === splitBySlice);
            if (splitField) {
                config.setRows(sliceFields);
                config.setColumns([splitField]);
                config.setValues(metricFields);
                console.log('Режим slices с разбивкой:', { sliceFields: sliceFields.length, columns: 1, values: metricFields.length });
            } else {
                // Fallback: обычный режим срезов
                config.setRows(sliceFields);
                config.setColumns([]);
                config.setValues(metricFields);
                console.log('Режим slices (fallback):', { sliceFields: sliceFields.length, columns: 0, values: metricFields.length });
            }
        } else {
            // Обычный режим срезов
            config.setRows(sliceFields);
            config.setColumns([]);
            config.setValues(metricFields);
            console.log('Режим slices:', { sliceFields: sliceFields.length, columns: 0, values: metricFields.length });
        }
    } else {
        // Обычный режим
        config.setRows(timeFields);
        config.setColumns(sliceFields);
        config.setValues(metricFields);
        console.log('Обычный режим:', { timeFields: timeFields.length, sliceFields: sliceFields.length, values: metricFields.length });
    }
    
    // Создаем фильтры для временных полей и срезов
    const filters = createFiltersFromMapping(mappingData);
    config.filters = filters;
    
    console.log('Финальная конфигурация:', {
        mode: config.mode,
        rows: config.rows.map(r => ({ name: r.name, type: r.type, level: r.level })),
        columns: config.columns.map(c => ({ name: c.name, type: c.type, level: c.level })),
        values: config.values.map(v => ({ name: v.name, type: v.type })),
        filters: config.filters.map(f => f.fieldName)
    });
    
    console.log('Конфигурация создана:', config);
    return config;
}

function renderNewPivotTable(rawData, mappingData, mode = 'normal', splitBySlice = '', originalMode = '') {
    console.log('Рендеринг новой сводной таблицы:', { 
        rawDataLength: rawData.length, 
        mappingData, 
        mode, 
        splitBySlice 
    });
    
    // Проверяем данные
    if (rawData && rawData.length > 0) {
        console.log('Первые 3 строки данных:', rawData.slice(0, 3));
        console.log('Структура данных:', Object.keys(rawData[0] || {}));
    } else {
        console.error('Нет данных для рендеринга!');
        return false;
    }
    
    try {
        // Создаем конфигурацию
        const config = createPivotConfigFromMapping(mappingData, mode, splitBySlice, originalMode);
        
        // Используем фильтры из глобальной переменной, если они есть
        if (window.currentFilters && window.currentFilters.length > 0) {
            config.filters = window.currentFilters;
            console.log('Используем активные фильтры:', config.filters.map(f => ({ 
                name: f.fieldName, 
                isActive: f.isActive, 
                type: f.fieldType 
            })));
        } else {
            // Инициализируем фильтры с данными
            config.filters.forEach(filter => {
                filter.getAvailableValues(rawData);
            });
        }
        
        // Создаем рендерер
        const renderer = new PivotRenderer('timeSeriesChartContainer');
        
        // Используем данные как есть - фильтрация уже применена в loadTimeSeriesData
        let dataToProcess = rawData;
        console.log(`Используем данные для обработки: ${dataToProcess.length} строк`);
        
        // Обрабатываем данные с учетом состояния коллапсирования
        const pivotData = new PivotData(dataToProcess);
        pivotData.process(config, renderer);
        
        // Сохраняем ссылки для перерисовки при коллапсировании
        window.currentPivotRenderer = renderer;
        window.currentPivotData = pivotData;
        window.currentPivotConfig = config;
        window.rawPivotData = rawData; // Сохраняем исходные данные для фильтров
        
        renderer.render(pivotData, config);
        
        console.log('Новая сводная таблица успешно отрендерена');
        
        // Автоматически создаем фильтры после рендеринга
        if (typeof createAutoFilters === 'function') {
            console.log('Вызываем createAutoFilters()...');
            try {
                createAutoFilters();
                console.log('createAutoFilters() вызвана успешно');
            } catch (error) {
                console.error('Ошибка в createAutoFilters():', error);
            }
        } else {
            console.log('Функция createAutoFilters не найдена');
        }
        
        return true;
    } catch (error) {
        console.error('Ошибка при рендеринге новой сводной таблицы:', error);
        console.error('Стек ошибки:', error.stack);
        return false;
    }
}

// Экспорт для использования в других файлах
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PivotField,
        PivotConfig,
        PivotData,
        PivotRenderer,
        createPivotConfigFromMapping,
        renderNewPivotTable
    };
}

// Делаем функции доступными в глобальной области видимости для браузера
if (typeof window !== 'undefined') {
    window.PivotField = PivotField;
    window.PivotFilter = PivotFilter;
    window.PivotConfig = PivotConfig;
    window.PivotData = PivotData;
    window.PivotRenderer = PivotRenderer;
    window.createPivotConfigFromMapping = createPivotConfigFromMapping;
    window.renderNewPivotTable = renderNewPivotTable;
    
    // Старые функции toggleTimeFieldCollapse и toggleSliceFieldCollapse удалены
    // Теперь используется triggerColumnButtons для триггера всех кнопок в столбце
    
    console.log('Новая система сводной таблицы загружена и доступна в глобальной области видимости');
    console.log('Доступные функции:', {
        PivotField: typeof PivotField,
        PivotConfig: typeof PivotConfig,
        PivotData: typeof PivotData,
        PivotRenderer: typeof PivotRenderer,
        createPivotConfigFromMapping: typeof createPivotConfigFromMapping,
        renderNewPivotTable: typeof renderNewPivotTable
    });
}

// Глобальная функция для переключения сортировки
window.togglePivotSort = function(fieldName, fieldType) {
    console.log('🎯🎯🎯 ФУНКЦИЯ togglePivotSort ВЫЗВАНА! 🎯🎯🎯');
    console.log('=== НАЧАЛО СОРТИРОВКИ ===');
    console.log('Переключение сортировки для поля:', fieldName, 'тип:', fieldType);
    console.log('currentPivotConfig:', !!window.currentPivotConfig);
    console.log('currentPivotRenderer:', !!window.currentPivotRenderer);
    console.log('currentPivotData:', !!window.currentPivotData);
    
    if (window.currentPivotConfig && window.currentPivotRenderer) {
        console.log('Старая конфигурация сортировки:', window.currentPivotConfig.sortConfig);
        
        // Переключаем сортировку в конфигурации
        window.currentPivotConfig.toggleSort(fieldName, fieldType);
        
        console.log('Новая конфигурация сортировки:', window.currentPivotConfig.sortConfig);
        
        // Применяем сортировку к данным
        if (window.currentPivotData) {
            console.log('🔍 Вызываем sortData...');
            console.log('Количество строк до сортировки:', window.currentPivotData.rowGroups.size);
            console.log('Примеры ключей ДО сортировки:', Array.from(window.currentPivotData.rowGroups.keys()).slice(0, 5));
            
            window.currentPivotData.sortData(window.currentPivotConfig.sortConfig);
            
            console.log('✅ sortData завершен');
            console.log('Количество строк после сортировки:', window.currentPivotData.rowGroups.size);
            console.log('Примеры ключей ПОСЛЕ сортировки:', Array.from(window.currentPivotData.rowGroups.keys()).slice(0, 5));
        } else {
            console.log('❌ window.currentPivotData отсутствует');
        }
        
        console.log('Перерисовываем таблицу...');
        // Перерисовываем таблицу
        window.currentPivotRenderer.render(window.currentPivotData, window.currentPivotConfig);
        
        console.log('Сортировка применена:', window.currentPivotConfig.sortConfig);
        console.log('=== КОНЕЦ СОРТИРОВКИ ===');
    } else {
        console.error('Нет активной конфигурации или рендерера для сортировки');
        console.log('currentPivotConfig:', window.currentPivotConfig);
        console.log('currentPivotRenderer:', window.currentPivotRenderer);
        console.log('currentPivotData:', window.currentPivotData);
    }
};

// Глобальная функция для переключения коллапсирования строк
function toggleRowCollapse(rowKey) {
    console.log('Переключение коллапсирования для строки:', rowKey);
    
    if (window.currentPivotRenderer) {
        window.currentPivotRenderer.toggleRowCollapse(rowKey);
    } else {
        console.error('Нет активного рендерера для коллапсирования строк');
    }
}

// Глобальная функция для триггера всех кнопок в столбце
function triggerColumnButtons(fieldName, fieldType) {
    console.log(`Триггер всех кнопок в столбце ${fieldName} (тип: ${fieldType})`);
    
    if (!window.currentPivotRenderer || !window.currentPivotConfig || !window.currentPivotData) {
        console.error('Нет активного рендерера или данных для триггера кнопок');
        return;
    }
    
    const renderer = window.currentPivotRenderer;
    const config = window.currentPivotConfig;
    const pivotData = window.currentPivotData;
    
    // Определяем уровень поля
    let visibleFields, fieldIndex;
    if (fieldType === 'time') {
        visibleFields = renderer.getVisibleTimeFields(config);
    } else if (fieldType === 'slice') {
        visibleFields = renderer.getVisibleSliceFields(config);
    } else {
        console.error('Неизвестный тип поля:', fieldType);
        return;
    }
    
    fieldIndex = visibleFields.findIndex(field => field.name === fieldName);
    if (fieldIndex === -1) {
        console.error('Поле не найдено в видимых полях');
        return;
    }
    
    console.log(`Находим все кнопки в столбце ${fieldName} (уровень ${fieldIndex})`);
    
    // Находим все кнопки в этом столбце
    const buttons = document.querySelectorAll(`.collapse-btn`);
    let triggeredCount = 0;
    
    buttons.forEach(button => {
        // Проверяем, находится ли кнопка в нужном столбце
        const row = button.closest('tr');
        if (!row) return;
        
        const cells = row.querySelectorAll('td');
        if (cells.length <= fieldIndex) return;
        
        const targetCell = cells[fieldIndex];
        if (targetCell.contains(button)) {
            console.log(`Нажатие кнопки в столбце ${fieldName}:`, button.textContent.trim());
            button.click();
            triggeredCount++;
        }
    });
    
    console.log(`Триггер завершен. Нажато кнопок: ${triggeredCount}`);
}

// Глобальная переменная для хранения экземпляра графика
let pivotChartInstance = null;

// Функция для построения графика из данных сводной таблицы
function updatePivotChart(chartDepthLevel = null) {
    console.log('🚀 === ПОСТРОЕНИЕ ГРАФИКА СВОДНОЙ ТАБЛИЦЫ ===');
    console.log('📊 chartDepthLevel:', chartDepthLevel);
    
    if (!window.currentPivotData || !window.currentPivotConfig) {
        console.error('Нет данных для построения графика');
        return;
    }
    
    const pivotData = window.currentPivotData;
    const config = window.currentPivotConfig;
    
    console.log('Данные для графика:', { 
        rowGroups: pivotData.rowGroups.size, 
        columnGroups: pivotData.columnGroups.size,
        mode: config.mode
    });
    
    // Определяем уровень вложенности для графика
    // Если не указан, используем максимальный уровень (самый детальный)
    const maxLevel = config.rows.length - 1;
    const targetLevel = chartDepthLevel !== null ? chartDepthLevel : maxLevel;
    
    console.log(`Уровень вложенности для графика: ${targetLevel} (макс: ${maxLevel})`);
    
    // Получаем данные для графика с учетом уровня вложенности
    const chartData = prepareChartData(pivotData, config, targetLevel);
    
    // Строим график
    renderPivotChart(chartData, config);
}

// Подготовка данных для графика
function prepareChartData(pivotData, config, targetLevel) {
    console.log('Подготовка данных для графика, уровень:', targetLevel);
    
    const labels = [];
    const datasets = [];
    
    // Получаем все ключи строк
    const allRowKeys = Array.from(pivotData.rowGroups.keys());
    
    // Фильтруем строки по уровню вложенности
    // Уровень = количество разделителей '|' в ключе
    const filteredRowKeys = allRowKeys.filter(key => {
        const level = key.split('|').length - 1;
        return level === targetLevel;
    });
    
    console.log(`Отфильтровано строк: ${filteredRowKeys.length} из ${allRowKeys.length}`);
    
    // Сортируем строки в хронологическом порядке (игнорируем сортировку таблицы)
    const sortedRowKeys = sortRowKeysChronologically(filteredRowKeys, config);
    
    console.log('Отсортированные ключи строк:', sortedRowKeys.slice(0, 10));
    
    // Создаем метки (labels) из ключей строк
    sortedRowKeys.forEach(rowKey => {
        const rowGroup = pivotData.rowGroups.get(rowKey);
        if (rowGroup) {
            // Формируем метку из полей строки
            const labelParts = [];
            config.rows.forEach(field => {
                if (rowGroup.fields[field.name] !== undefined) {
                    labelParts.push(rowGroup.fields[field.name]);
                }
            });
            labels.push(labelParts.join(' | '));
        }
    });
    
    // Создаем datasets для каждой метрики и столбца
    if (config.mode === 'split-columns') {
        // Режим разбивки по столбцам: создаем dataset для каждого столбца
        const columnKeys = Array.from(pivotData.columnGroups.keys());
        
        config.values.forEach(valueField => {
            columnKeys.forEach(colKey => {
                const dataValues = [];
                
                sortedRowKeys.forEach(rowKey => {
                    // Получаем значение для этой строки и столбца
                    let value = 0;
                    
                    // Проверяем, есть ли эта строка в crossTable (детализированная строка)
                    if (pivotData.crossTable.has(rowKey)) {
                        value = pivotData.getValue(rowKey, colKey, valueField.name);
                    } else {
                        // Агрегированная строка - суммируем значения из всех дочерних элементов
                        const allKeys = Array.from(pivotData.rowGroups.keys());
                        allKeys.forEach(childKey => {
                            if (childKey.startsWith(rowKey + '|')) {
                                value += pivotData.getValue(childKey, colKey, valueField.name);
                            }
                        });
                    }
                    
                    dataValues.push(value);
                });
                
                // Получаем информацию о столбце для метки
                const columnGroup = pivotData.columnGroups.get(colKey);
                const columnLabel = columnGroup ? columnGroup.fields[config.columns[0].name] : colKey;
                
                datasets.push({
                    label: `${valueField.label} - ${columnLabel}`,
                    data: dataValues,
                    borderWidth: 2,
                    fill: false
                });
            });
        });
    } else {
        // Обычный режим: создаем dataset для каждой метрики
        config.values.forEach(valueField => {
            const dataValues = [];
            
            sortedRowKeys.forEach(rowKey => {
                const rowGroup = pivotData.rowGroups.get(rowKey);
                if (rowGroup && rowGroup.rows && rowGroup.rows.length > 0) {
                    // Суммируем значения из всех строк в группе
                    let sum = 0;
                    rowGroup.rows.forEach(row => {
                        const value = parseFloat(row[valueField.name]) || 0;
                        sum += value;
                    });
                    dataValues.push(sum);
                } else {
                    dataValues.push(0);
                }
            });
            
            datasets.push({
                label: valueField.label,
                data: dataValues,
                borderWidth: 2,
                fill: false
            });
        });
    }
    
    return { labels, datasets };
}

// Сортировка ключей строк в хронологическом порядке
function sortRowKeysChronologically(rowKeys, config) {
    console.log('Сортировка строк в хронологическом порядке');
    
    // Если это временные ряды, сортируем по временным полям
    const timeFields = config.rows.filter(field => field.type === 'time');
    
    if (timeFields.length > 0) {
        return rowKeys.sort((a, b) => {
            const fieldsA = a.split('|');
            const fieldsB = b.split('|');
            
            // Сравниваем по каждому временному полю
            for (let i = 0; i < Math.min(fieldsA.length, fieldsB.length); i++) {
                const valueA = fieldsA[i];
                const valueB = fieldsB[i];
                
                // Пробуем преобразовать в число для сравнения
                const numA = parseFloat(valueA);
                const numB = parseFloat(valueB);
                
                if (!isNaN(numA) && !isNaN(numB)) {
                    if (numA !== numB) return numA - numB;
                } else {
                    // Текстовое сравнение
                    if (valueA < valueB) return -1;
                    if (valueA > valueB) return 1;
                }
            }
            
            return 0;
        });
    }
    
    // Для срезов используем существующий порядок
    return rowKeys;
}

// Рендеринг графика
function renderPivotChart(chartData, config) {
    console.log('🎯 === НАЧАЛО РЕНДЕРИНГА ГРАФИКА ===');
    console.log('📊 Данные графика:', chartData);
    console.log('⚙️ Конфигурация:', config);
    
    const canvas = document.getElementById('pivotChart');
    if (!canvas) {
        console.error('Canvas для графика не найден');
        return;
    }
    
    // Уничтожаем предыдущий экземпляр графика
    if (pivotChartInstance) {
        pivotChartInstance.destroy();
    }
    
    // Генерируем цвета для datasets
    const colors = generateChartColors(chartData.datasets.length);
    chartData.datasets.forEach((dataset, index) => {
        dataset.borderColor = colors[index];
        dataset.backgroundColor = colors[index] + '33'; // Добавляем прозрачность
        dataset.borderWidth = 3; // Увеличиваем толщину линии
        dataset.pointRadius = 4; // Размер точек
        dataset.pointHoverRadius = 6; // Размер точек при наведении
        dataset.pointBackgroundColor = colors[index];
        dataset.pointBorderColor = '#fff';
        dataset.pointBorderWidth = 2;
        dataset.tension = 0.1; // Небольшое скругление линий
    });
    
    // Находим минимальное и максимальное значения для настройки оси Y
    let allValues = [];
    chartData.datasets.forEach(dataset => {
        allValues = allValues.concat(dataset.data.filter(v => v > 0)); // Игнорируем нули
    });
    
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const range = maxValue - minValue;
    
    // Подход Google Sheets: более агрессивные отступы (30% от диапазона)
    const padding = Math.max(range * 0.3, (maxValue * 0.15)); // Минимум 15% от максимального значения
    
    // Google всегда показывает значительную часть нулевой области
    const shouldStartFromZero = minValue < (maxValue * 0.1); // Увеличили порог до 10%
    const yMin = shouldStartFromZero ? 0 : Math.max(0, minValue - padding);
    const yMax = maxValue + padding;
    
    // Подход Google Sheets: вычисляем оптимальный шаг для оси Y (стремимся к 6-8 делениям)
    const targetSteps = 6; // Меньше делений для лучшей читаемости
    const actualRange = yMax - yMin; // Используем полный диапазон с отступами
    const rawStepSize = actualRange / targetSteps;
    
    // Более агрессивное округление шага до красивого числа (как в Google Sheets)
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStepSize)));
    const residual = rawStepSize / magnitude;
    let stepSize;
    if (residual > 7) {
        stepSize = 10 * magnitude;
    } else if (residual > 5) {
        stepSize = 7.5 * magnitude;
    } else if (residual > 3) {
        stepSize = 5 * magnitude;
    } else if (residual > 2) {
        stepSize = 2.5 * magnitude;
    } else if (residual > 1) {
        stepSize = 2 * magnitude;
    } else {
        stepSize = magnitude;
    }
    
    console.log('📈 === НАСТРОЙКИ ОСИ Y ===');
    console.log('🔢 Диапазон значений:', { 
        minValue, 
        maxValue, 
        range, 
        padding,
        shouldStartFromZero,
        yMin, 
        yMax, 
        stepSize,
        expectedSteps: Math.ceil(range / stepSize),
        actualRange: yMax - yMin
    });
    console.log('🎛️ Параметры Chart.js для оси Y:', {
        beginAtZero: shouldStartFromZero,
        min: shouldStartFromZero ? undefined : yMin,
        max: yMax,
        stepSize: stepSize
    });
    
    // Функция для форматирования больших чисел
    function formatLargeNumber(value) {
        if (value >= 1000000000) {
            return (value / 1000000000).toFixed(1) + ' млрд';
        } else if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + ' млн';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + ' тыс';
        }
        return value.toFixed(0);
    }
    
    // Создаем новый график
    const ctx = canvas.getContext('2d');
    pivotChartInstance = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: config.mode === 'split-columns' ? 'Временные ряды с разбивкой' : 'Временные ряды',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toLocaleString('ru-RU');
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Период'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Значение',
                        font: { size: 14, weight: 'bold' }
                    },
                    beginAtZero: shouldStartFromZero,
                    min: shouldStartFromZero ? undefined : yMin, // Если начинаем с нуля, не задаем min
                    max: yMax,
                    ticks: {
                        stepSize: stepSize,
                        maxTicksLimit: 8, // Меньше делений для читаемости
                        callback: function(value) {
                            return formatLargeNumber(value);
                        },
                        font: { size: 12 },
                        padding: 8 // Больше отступы между метками и осью
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.08)', // Более тонкие линии сетки
                        lineWidth: 1,
                        drawBorder: false // Убираем границу оси
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
    
    console.log('График успешно создан');
}

// Генерация цветов для графика
function generateChartColors(count) {
    const baseColors = [
        '#007bff', // Синий
        '#28a745', // Зеленый
        '#dc3545', // Красный
        '#ffc107', // Желтый
        '#17a2b8', // Голубой
        '#6f42c1', // Фиолетовый
        '#fd7e14', // Оранжевый
        '#20c997', // Бирюзовый
        '#e83e8c', // Розовый
        '#6c757d'  // Серый
    ];
    
    const colors = [];
    for (let i = 0; i < count; i++) {
        colors.push(baseColors[i % baseColors.length]);
    }
    
    return colors;
}

// Проверяем, что функции доступны глобально
console.log('🔍 Проверка доступности функций:');
console.log('- window.togglePivotSort:', typeof window.togglePivotSort);
console.log('- togglePivotSort:', typeof togglePivotSort);
console.log('- window.updatePivotChart:', typeof window.updatePivotChart);
console.log('- updatePivotChart:', typeof updatePivotChart);
