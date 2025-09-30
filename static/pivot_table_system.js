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
    
    process(config) {
        console.log('Обработка данных сводной таблицы с конфигурацией:', config);
        
        // Группируем данные по строкам и столбцам
        this.groupByRowsAndColumns(config);
        
        // Создаем перекрестную таблицу
        this.createCrossTable(config);
        
        return this;
    }
    
    groupByRowsAndColumns(config) {
        this.rowGroups.clear();
        this.columnGroups.clear();
        
        // Группировка по строкам
        this.rawData.forEach(row => {
            const rowKey = this.createRowKey(row, config.rows);
            if (!this.rowGroups.has(rowKey)) {
                this.rowGroups.set(rowKey, {
                    key: rowKey,
                    fields: this.extractFieldValues(row, config.rows),
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
            columnGroups: this.columnGroups.size
        });
    }
    
    createCrossTable(config) {
        this.crossTable.clear();
        
        console.log('Создание перекрестной таблицы:', {
            rawDataLength: this.rawData.length,
            rows: config.rows.map(r => r.name),
            columns: config.columns.map(c => c.name),
            values: config.values.map(v => v.name)
        });
        
        // Создаем перекрестную таблицу: rowKey -> colKey -> aggregatedValue
        this.rawData.forEach((row, index) => {
            if (index < 3) { // Логируем первые 3 строки
                console.log(`Строка ${index}:`, row);
            }
            
            const rowKey = this.createRowKey(row, config.rows);
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
    
    getRowFields(rowKey) {
        return this.rowGroups.get(rowKey)?.fields || {};
    }
    
    getColumnFields(colKey) {
        return this.columnGroups.get(colKey)?.fields || {};
    }
}

class PivotRenderer {
    constructor(containerId) {
        this.containerId = containerId;
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
            config.rows.forEach(rowField => {
                html += `<th class="pivot-header" rowspan="2">${rowField.label}</th>`;
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
            
            // Заголовки для временных полей (строки)
            config.rows.forEach(rowField => {
                html += `<th class="pivot-header">${rowField.label}</th>`;
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
        
        if (config.mode === 'split-columns') {
            // Режим разбивки по столбцам
            const rowKeys = pivotData.getRowKeys();
            const columnKeys = pivotData.getColumnKeys();
            
            rowKeys.forEach(rowKey => {
                html += '<tr>';
                
                // Значения строк (временные поля)
                const rowFields = pivotData.getRowFields(rowKey);
                config.rows.forEach(rowField => {
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
                
                // Значения временных полей (строки)
                const rowFields = pivotData.getRowFields(rowKey);
                config.rows.forEach(rowField => {
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
}

// Функции для работы с новой системой
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
    
    console.log('Финальная конфигурация:', {
        rows: config.rows.map(r => r.name),
        columns: config.columns.map(c => c.name),
        values: config.values.map(v => v.name)
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
        
        // Обрабатываем данные
        const pivotData = new PivotData(rawData);
        pivotData.process(config);
        
        // Рендерим таблицу
        const renderer = new PivotRenderer('timeSeriesChartContainer');
        renderer.render(pivotData, config);
        
        console.log('Новая сводная таблица успешно отрендерена');
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
    window.PivotConfig = PivotConfig;
    window.PivotData = PivotData;
    window.PivotRenderer = PivotRenderer;
    window.createPivotConfigFromMapping = createPivotConfigFromMapping;
    window.renderNewPivotTable = renderNewPivotTable;
    
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
