/**
 * Модуль для работы с фильтрами сводной таблицы
 * Версия: 1.0.0
 */

class PivotFiltersModule {
    constructor() {
        this.filters = [];
        this.filtersContainer = null;
        this.onFilterChange = null;
    }

    /**
     * Инициализация модуля фильтров
     * @param {string} containerId - ID контейнера для фильтров
     * @param {Function} onFilterChange - Callback при изменении фильтров
     */
    init(containerId, onFilterChange) {
        this.filtersContainer = document.getElementById(containerId);
        this.onFilterChange = onFilterChange;
        
        if (!this.filtersContainer) {
            console.error('❌ Контейнер фильтров не найден:', containerId);
            return false;
        }
        
        console.log('✅ Модуль фильтров инициализирован');
        return true;
    }

    /**
     * Создание фильтров на основе данных и маппинга
     * @param {Array} data - Массив данных
     * @param {Object} mapping - Конфигурация маппинга
     * @returns {Array} Массив созданных фильтров
     */
    createFilters(data, mapping) {
        if (!this.filtersContainer) {
            console.error('❌ Модуль не инициализирован');
            return [];
        }

        if (!data || !mapping || !mapping.columns) {
            console.warn('⚠️ Нет данных или маппинга для создания фильтров');
            return [];
        }

        console.log('🔧 Создание фильтров для', data.length, 'строк');

        // Очищаем контейнер
        this.filtersContainer.innerHTML = '';
        this.filters = [];

        // Получаем dimension-колонки
        const dimensionColumns = mapping.columns.filter(col => 
            col.role === 'dimension' && col.include !== false
        );

        console.log('📊 Найдено dimension-колонок:', dimensionColumns.length);

        // Создаем фильтр для каждой dimension-колонки
        dimensionColumns.forEach(col => {
            const filter = this._createFilter(col.name, col.type, data);
            if (filter) {
                this.filters.push(filter);
            }
        });

        console.log('✅ Создано фильтров:', this.filters.length);
        return this.filters;
    }

    /**
     * Создание отдельного фильтра
     * @private
     */
    _createFilter(columnName, columnType, data) {
        // Получаем уникальные значения
        const uniqueValues = [...new Set(data.map(row => row[columnName]))]
            .filter(v => v !== null && v !== undefined && v !== '')
            .sort((a, b) => {
                if (columnType === 'numeric') {
                    return Number(a) - Number(b);
                }
                return String(a).localeCompare(String(b));
            });

        if (uniqueValues.length === 0) {
            console.log('⚠️ Нет значений для фильтра:', columnName);
            return null;
        }

        console.log(`📌 Создание фильтра "${columnName}":`, uniqueValues.length, 'значений');

        // Создаем HTML для фильтра
        const filterDiv = document.createElement('div');
        filterDiv.className = 'filter-card';
        filterDiv.dataset.filterName = columnName;

        const maxVisible = 20;
        const hasMore = uniqueValues.length > maxVisible;
        const visibleValues = uniqueValues.slice(0, maxVisible);

        filterDiv.innerHTML = `
            <div class="filter-header">
                <span>${columnName}</span>
                <button class="btn btn-sm btn-link text-decoration-none" onclick="pivotFiltersModule.toggleFilter('${columnName}')">
                    <i class="fas fa-chevron-down" id="filter-icon-${columnName}"></i>
                </button>
            </div>
            <div class="filter-body" id="filter-body-${columnName}">
                <div class="mb-2">
                    <button class="btn btn-sm btn-outline-secondary me-1" onclick="pivotFiltersModule.selectAll('${columnName}')">
                        Выбрать все
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="pivotFiltersModule.deselectAll('${columnName}')">
                        Снять все
                    </button>
                </div>
                <div class="filter-options" id="filter-options-${columnName}">
                    ${visibleValues.map(value => `
                        <div class="filter-option">
                            <input type="checkbox" 
                                   class="form-check-input" 
                                   id="filter-${columnName}-${this._sanitizeId(value)}" 
                                   value="${this._escapeHtml(value)}" 
                                   data-filter="${columnName}"
                                   checked 
                                   onchange="pivotFiltersModule.onFilterValueChange()">
                            <label class="form-check-label" for="filter-${columnName}-${this._sanitizeId(value)}">
                                ${this._escapeHtml(value)}
                            </label>
                        </div>
                    `).join('')}
                    ${hasMore ? `
                        <div class="text-muted small mt-2">
                            <i class="fas fa-info-circle me-1"></i>
                            Показано ${maxVisible} из ${uniqueValues.length} значений
                            <button class="btn btn-sm btn-link p-0 ms-2" onclick="pivotFiltersModule.showAllValues('${columnName}', ${JSON.stringify(uniqueValues).replace(/"/g, '&quot;')})">
                                Показать все
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        this.filtersContainer.appendChild(filterDiv);

        return {
            field: columnName,
            type: columnType,
            values: uniqueValues,
            allValues: uniqueValues
        };
    }

    /**
     * Переключение видимости фильтра
     */
    toggleFilter(columnName) {
        const body = document.getElementById(`filter-body-${columnName}`);
        const icon = document.getElementById(`filter-icon-${columnName}`);
        
        if (body && icon) {
            const isCollapsed = body.style.display === 'none';
            body.style.display = isCollapsed ? 'block' : 'none';
            icon.className = isCollapsed ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        }
    }

    /**
     * Выбрать все значения фильтра
     */
    selectAll(columnName) {
        const checkboxes = document.querySelectorAll(`input[data-filter="${columnName}"]`);
        checkboxes.forEach(cb => cb.checked = true);
        this.onFilterValueChange();
    }

    /**
     * Снять все значения фильтра
     */
    deselectAll(columnName) {
        const checkboxes = document.querySelectorAll(`input[data-filter="${columnName}"]`);
        checkboxes.forEach(cb => cb.checked = false);
        this.onFilterValueChange();
    }

    /**
     * Показать все значения фильтра
     */
    showAllValues(columnName, allValues) {
        const optionsContainer = document.getElementById(`filter-options-${columnName}`);
        if (!optionsContainer) return;

        optionsContainer.innerHTML = allValues.map(value => `
            <div class="filter-option">
                <input type="checkbox" 
                       class="form-check-input" 
                       id="filter-${columnName}-${this._sanitizeId(value)}" 
                       value="${this._escapeHtml(value)}" 
                       data-filter="${columnName}"
                       checked 
                       onchange="pivotFiltersModule.onFilterValueChange()">
                <label class="form-check-label" for="filter-${columnName}-${this._sanitizeId(value)}">
                    ${this._escapeHtml(value)}
                </label>
            </div>
        `).join('');
    }

    /**
     * Callback при изменении значения фильтра
     */
    onFilterValueChange() {
        if (this.onFilterChange) {
            this.onFilterChange();
        }
    }

    /**
     * Получение отфильтрованных данных
     * @param {Array} data - Исходные данные
     * @returns {Array} Отфильтрованные данные
     */
    getFilteredData(data) {
        if (!data || data.length === 0) {
            return [];
        }

        let filtered = [...data];

        this.filters.forEach(filter => {
            const selectedValues = this.getSelectedValues(filter.field);
            
            // Если выбраны не все значения, применяем фильтр
            if (selectedValues.length > 0 && selectedValues.length < filter.values.length) {
                filtered = filtered.filter(row => {
                    const rowValue = String(row[filter.field]);
                    return selectedValues.includes(rowValue);
                });
            }
        });

        console.log(`🔍 Фильтрация: ${data.length} → ${filtered.length} строк`);
        return filtered;
    }

    /**
     * Получение выбранных значений для фильтра
     * @param {string} columnName - Название колонки
     * @returns {Array} Массив выбранных значений
     */
    getSelectedValues(columnName) {
        const checkboxes = document.querySelectorAll(`input[data-filter="${columnName}"]:checked`);
        return Array.from(checkboxes).map(cb => cb.value);
    }

    /**
     * Сброс всех фильтров
     */
    resetAll() {
        const checkboxes = this.filtersContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = true);
        this.onFilterValueChange();
    }

    /**
     * Получение состояния фильтров
     * @returns {Object} Объект с состоянием фильтров
     */
    getState() {
        const state = {};
        this.filters.forEach(filter => {
            state[filter.field] = this.getSelectedValues(filter.field);
        });
        return state;
    }

    /**
     * Восстановление состояния фильтров
     * @param {Object} state - Объект с состоянием фильтров
     */
    setState(state) {
        if (!state) return;

        Object.keys(state).forEach(field => {
            const selectedValues = state[field];
            const checkboxes = document.querySelectorAll(`input[data-filter="${field}"]`);
            
            checkboxes.forEach(cb => {
                cb.checked = selectedValues.includes(cb.value);
            });
        });

        this.onFilterValueChange();
    }

    /**
     * Вспомогательные функции
     */
    _sanitizeId(value) {
        return String(value).replace(/[^a-zA-Z0-9-_]/g, '-');
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Создаем глобальный экземпляр
window.pivotFiltersModule = new PivotFiltersModule();
console.log('✅ Модуль pivot_filters.js загружен');

