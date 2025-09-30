// ========================================
// НОВАЯ СИСТЕМА СВОДНОЙ ТАБЛИЦЫ
// ========================================

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
        if (!this.isActive) return true;
        
        const value = row[this.fieldName];
        if (value === null || value === undefined) return false;
        
        if (this.fieldType === 'text') {
            return this.values.includes(value);
        } else if (this.fieldType === 'number') {
            return value >= this.minValue && value <= this.maxValue;
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
        
        // Получаем видимые временные поля для группировки строк
        const visibleRowFields = renderer ? renderer.getVisibleTimeFields(config) : config.rows;
        
        console.log('Группировка строк по полям:', visibleRowFields.map(f => f.name));
        
        // Группировка по строкам (только по видимым временным полям)
        this.rawData.forEach(row => {
            const rowKey = this.createRowKey(row, visibleRowFields);
            if (!this.rowGroups.has(rowKey)) {
                this.rowGroups.set(rowKey, {
                    key: rowKey,
                    fields: this.extractFieldValues(row, visibleRowFields),
                    rows: []
                });
            }
            this.rowGroups.get(rowKey).rows.push(row);
        });
        
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
        
        // Получаем видимые поля для создания ключей
        const visibleRowFields = renderer ? renderer.getVisibleTimeFields(config) : config.rows;
        
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
        
        console.log('Перекрестная таблица создана:', {
            rows: this.crossTable.size,
            totalCells: Array.from(this.crossTable.values()).reduce((sum, colMap) => sum + colMap.size, 0)
        });
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
    
    getRowKeys() {
        return Array.from(this.rowGroups.keys()).sort();
    }
    
    getColumnKeys() {
        return Array.from(this.columnGroups.keys()).sort();
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
    }
    
    render(pivotData, config) {
        console.log('Рендеринг сводной таблицы:', { config, pivotData });
        
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error('Контейнер не найден:', this.containerId);
            return;
        }
        
        // Создаем HTML для сводной таблицы
        const html = this.createPivotTableHTML(pivotData, config);
        container.innerHTML = html;
        
        console.log('Сводная таблица отрендерена');
    }
    
    createPivotTableHTML(pivotData, config) {
        let html = '<div class="pivot-table-container">';
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
    
    createHeadersHTML(pivotData, config) {
        let html = '<thead class="table-dark">';
        
        if (config.mode === 'split-columns') {
            // Режим разбивки по столбцам (как в Google Sheets)
            
            // Первый уровень заголовков - метрики
            html += '<tr>';
            
            // Заголовки для временных полей (строки) с поддержкой коллапсирования
            const visibleTimeFields = this.getVisibleTimeFields(config);
            visibleTimeFields.forEach(rowField => {
                const isCollapsible = this.hasChildTimeFields(config, rowField);
                const isCollapsed = this.isTimeFieldCollapsed(rowField.name);
                const collapseIcon = isCollapsible ? 
                    `<span class="collapse-icon" onclick="toggleTimeFieldCollapse('${rowField.name}')" style="cursor: pointer; margin-right: 5px;">${isCollapsed ? '+' : '−'}</span>` : 
                    '<span style="margin-right: 12px;"></span>';
                
                html += `<th class="pivot-header time-header" data-field="${rowField.name}" data-level="${rowField.level}" rowspan="2">${collapseIcon}${rowField.label}</th>`;
            });
            
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                html += `<th class="pivot-header metric-header" colspan="${columnKeys.length}">${valueField.label}</th>`;
            });
            html += '</tr>';
            
            // Второй уровень заголовков - срезы
            html += '<tr>';
            config.values.forEach(valueField => {
                const columnKeys = pivotData.getColumnKeys();
                columnKeys.forEach(colKey => {
                    const colFields = pivotData.getColumnFields(colKey);
                    const colLabel = config.columns.map(colField => colFields[colField.name]).join(' - ');
                    html += `<th class="pivot-header slice-header">${colLabel || colKey}</th>`;
                });
            });
            html += '</tr>';
        } else if (config.mode === 'time-series') {
            // Режим временных рядов
            html += '<tr>';
            
            // Заголовки для видимых временных полей (строки) с поддержкой коллапсирования
            const visibleTimeFields = this.getVisibleTimeFields(config);
            visibleTimeFields.forEach(rowField => {
                const isCollapsible = this.hasChildTimeFields(config, rowField);
                const isCollapsed = this.isTimeFieldCollapsed(rowField.name);
                const collapseIcon = isCollapsible ? 
                    `<span class="collapse-icon" onclick="toggleTimeFieldCollapse('${rowField.name}')" style="cursor: pointer; margin-right: 5px;">${isCollapsed ? '+' : '−'}</span>` : 
                    '<span style="margin-right: 12px;"></span>';
                
                html += `<th class="pivot-header time-header" data-field="${rowField.name}" data-level="${rowField.level}">${collapseIcon}${rowField.label}</th>`;
            });
            
            // Заголовки для метрик (значения)
            config.values.forEach(valueField => {
                html += `<th class="pivot-header">${valueField.label}</th>`;
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
            // Режим разбивки по столбцам
            const rowKeys = pivotData.getRowKeys();
            const columnKeys = pivotData.getColumnKeys();
            
            rowKeys.forEach(rowKey => {
                html += '<tr>';
                
                // Значения видимых временных полей (строки)
                const visibleTimeFields = this.getVisibleTimeFields(config);
                const rowFields = pivotData.getRowFields(rowKey, visibleTimeFields);
                visibleTimeFields.forEach(rowField => {
                    html += `<td class="pivot-cell">${rowFields[rowField.name] || ''}</td>`;
                });
                
                // Значения для каждой метрики и каждого столбца (среза)
                config.values.forEach(valueField => {
                    columnKeys.forEach(colKey => {
                        const value = pivotData.getValue(rowKey, colKey, valueField.name);
                        html += `<td class="pivot-cell" style="text-align: right;">${this.formatValue(value)}</td>`;
                    });
                });
                
                html += '</tr>';
            });
        } else if (config.mode === 'time-series') {
            // Режим временных рядов
            const rowKeys = pivotData.getRowKeys();
            
            rowKeys.forEach(rowKey => {
                html += '<tr>';
                
                // Значения видимых временных полей (строки)
                const rowFields = pivotData.getRowFields(rowKey);
                const visibleTimeFields = this.getVisibleTimeFields(config);
                visibleTimeFields.forEach(rowField => {
                    html += `<td class="pivot-cell">${rowFields[rowField.name] || ''}</td>`;
                });
                
                // Значения метрик - агрегируем по всем строкам в группе
                config.values.forEach(valueField => {
                    const rowGroup = pivotData.rowGroups.get(rowKey);
                    let aggregatedValue = 0;
                    
                    // Суммируем все значения метрики в группе строк
                    if (rowGroup && rowGroup.rows) {
                        console.log(`Агрегация для ${rowKey}, метрика ${valueField.name}:`, {
                            rowCount: rowGroup.rows.length,
                            rows: rowGroup.rows.slice(0, 3) // Показываем первые 3 строки
                        });
                        
                        rowGroup.rows.forEach((row, index) => {
                            const value = parseFloat(row[valueField.name]) || 0;
                            aggregatedValue += value;
                            
                            if (index < 3) { // Логируем первые 3 значения
                                console.log(`  Строка ${index}: ${row[valueField.name]} -> ${value}`);
                            }
                        });
                        
                        console.log(`Итого для ${rowKey}: ${aggregatedValue}`);
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
            const visibleTimeFields = this.getVisibleTimeFields(config);
            const columnKeys = pivotData.getColumnKeys();
            const totals = pivotData.calculateTotals(config);
            
            // Ячейки для временных полей с "Total" в первой
            visibleTimeFields.forEach((field, index) => {
                const cellContent = index === 0 ? '<strong>Total</strong>' : '';
                html += `<td class="pivot-cell pivot-total-cell">${cellContent}</td>`;
            });
            
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
    
    isTimeFieldCollapsed(fieldName) {
        return this.collapsedTimeFields.has(fieldName);
    }
    
    toggleTimeFieldCollapse(fieldName) {
        if (this.collapsedTimeFields.has(fieldName)) {
            this.collapsedTimeFields.delete(fieldName);
        } else {
            this.collapsedTimeFields.add(fieldName);
        }
        console.log('Переключено состояние коллапсирования для поля:', fieldName, 'Свернуто:', this.collapsedTimeFields.has(fieldName));
    }
    
    getVisibleTimeFields(config) {
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
            }
            
            const filter = new PivotFilter(col.name, fieldType, col.name);
            filters.push(filter);
            console.log('Создан фильтр:', { name: col.name, type: fieldType });
        }
    });
    
    return filters;
}

function createPivotConfigFromMapping(mappingData, mode = 'normal', splitBySlice = '') {
    console.log('Создание конфигурации сводной таблицы из маппинга:', { mappingData, mode, splitBySlice });
    
    const config = new PivotConfig();
    config.setMode(mode);
    
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
        metricFields: metricFields.length
    });
    
    // Сортируем временные поля по уровню
    timeFields.sort((a, b) => a.level - b.level);
    
    // Конфигурируем в зависимости от режима
    if (mode === 'split-columns' && splitBySlice) {
        // Режим разбивки по столбцам
        config.setRows(timeFields);
        config.setColumns([new PivotField(splitBySlice, splitBySlice, 'slice')]);
        config.setValues(metricFields);
        console.log('Режим split-columns:', { timeFields: timeFields.length, columns: 1, values: metricFields.length });
    } else if (mode === 'time-series') {
        // Режим временных рядов
        config.setRows(timeFields);
        config.setColumns([]);
        config.setValues(metricFields);
        console.log('Режим time-series:', { timeFields: timeFields.length, columns: 0, values: metricFields.length });
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
        rows: config.rows.map(r => r.name),
        columns: config.columns.map(c => c.name),
        values: config.values.map(v => v.name),
        filters: config.filters.map(f => f.fieldName)
    });
    
    console.log('Конфигурация создана:', config);
    return config;
}

function renderNewPivotTable(rawData, mappingData, mode = 'normal', splitBySlice = '') {
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
        const config = createPivotConfigFromMapping(mappingData, mode, splitBySlice);
        
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
        
        // Применяем фильтры к данным перед обработкой
        let dataToProcess = rawData;
        if (config.filters && config.filters.length > 0) {
            const activeFilters = config.filters.filter(f => f.isActive);
            if (activeFilters.length > 0) {
                console.log('Применяем фильтры к данным перед обработкой:', activeFilters.map(f => f.fieldName));
                dataToProcess = rawData.filter(row => {
                    return activeFilters.every(filter => filter.matches(row));
                });
                console.log(`Фильтрация данных: ${rawData.length} -> ${dataToProcess.length} строк`);
            }
        }
        
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
    
    // Глобальная функция для переключения состояния коллапсирования
    window.toggleTimeFieldCollapse = function(fieldName) {
        console.log('Переключение коллапсирования для поля:', fieldName);
        
        // Находим активный рендерер (если есть)
        if (window.currentPivotRenderer) {
            window.currentPivotRenderer.toggleTimeFieldCollapse(fieldName);
            
            // Переобрабатываем данные с учетом нового состояния коллапсирования
            if (window.currentPivotData && window.currentPivotConfig) {
                // Получаем исходные данные
                const rawData = window.currentPivotData.rawData;
                
                // Создаем новые данные с обновленной группировкой
                const newPivotData = new PivotData(rawData);
                newPivotData.process(window.currentPivotConfig, window.currentPivotRenderer);
                
                // Обновляем ссылку на данные
                window.currentPivotData = newPivotData;
                
                // Перерисовываем таблицу
                window.currentPivotRenderer.render(newPivotData, window.currentPivotConfig);
            }
        }
    };
    
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
