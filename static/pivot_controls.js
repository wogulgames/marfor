/**
 * Модуль для управления интерфейсом сводной таблицы
 * Версия: 1.0.0
 */

class PivotControlsModule {
    constructor() {
        this.state = {
            currentMode: 'time-series',
            selectedMetrics: [],
            selectedSlices: [],
            selectedSplitField: ''
        };
        
        this.elements = {};
        this.dataMapping = null;
        this.onStateChange = null;
    }

    /**
     * Инициализация модуля
     * @param {Object} config - Конфигурация
     */
    init(config) {
        this.dataMapping = config.dataMapping;
        this.onStateChange = config.onStateChange;

        // Находим элементы интерфейса
        this.elements = {
            timeSeriesModeBtn: document.getElementById('timeSeriesModeBtn'),
            slicesModeBtn: document.getElementById('slicesModeBtn'),
            pivotModeTitle: document.getElementById('pivotModeTitle'),
            primaryFieldLabel: document.getElementById('primaryFieldLabel'),
            primaryFieldDropdownText: document.getElementById('primaryFieldDropdownText'),
            primaryFieldCheckboxes: document.getElementById('primaryFieldCheckboxes'),
            splitFieldLabel: document.getElementById('splitFieldLabel'),
            splitFieldDropdownText: document.getElementById('splitFieldDropdownText'),
            splitFieldRadioButtons: document.getElementById('splitFieldRadioButtons'),
            noSplitFieldText: document.getElementById('noSplitFieldText')
        };

        // Проверяем наличие элементов
        const missingElements = Object.keys(this.elements).filter(key => !this.elements[key]);
        if (missingElements.length > 0) {
            console.error('❌ Не найдены элементы:', missingElements);
            return false;
        }

        console.log('✅ Модуль управления инициализирован');
        return true;
    }

    /**
     * Заполнение селектов на основе режима
     */
    populateSelects() {
        if (!this.dataMapping || !this.dataMapping.columns) {
            console.error('❌ Нет маппинга для заполнения селектов');
            return;
        }

        console.log('🚀 Заполнение селектов, режим:', this.state.currentMode);

        // Очищаем селекты
        this.elements.primaryFieldCheckboxes.innerHTML = '';
        this.elements.splitFieldRadioButtons.innerHTML = '';

        let availablePrimaryFields = [];
        let availableSplitFields = [];

        if (this.state.currentMode === 'time-series') {
            // Режим временных рядов
            availablePrimaryFields = this.dataMapping.columns.filter(col => 
                col.role === 'metric' && col.include !== false
            );
            availableSplitFields = this.dataMapping.columns.filter(col => 
                col.role === 'dimension' && !col.time_series && col.include !== false
            );

            this.elements.primaryFieldLabel.textContent = 'Метрики';
            this.elements.splitFieldLabel.textContent = 'Разбивка по столбцам';
            this.elements.noSplitFieldText.textContent = 'Только временные ряды';
        } else {
            // Режим срезов
            availablePrimaryFields = this.dataMapping.columns.filter(col => 
                col.role === 'dimension' && !col.time_series && col.include !== false
            );
            availableSplitFields = this.dataMapping.columns.filter(col => 
                col.time_series && col.include !== false
            );

            this.elements.primaryFieldLabel.textContent = 'Срезы';
            this.elements.splitFieldLabel.textContent = 'Разбивка по времени';
            this.elements.noSplitFieldText.textContent = 'Только срезы';
        }

        console.log('📊 Основных полей:', availablePrimaryFields.length);
        console.log('📊 Полей для разбивки:', availableSplitFields.length);

        // Заполняем чекбоксы основного поля
        availablePrimaryFields.forEach(field => {
            const li = document.createElement('li');
            li.className = 'dropdown-item-text';
            li.innerHTML = `
                <div class="form-check">
                    <input class="form-check-input" 
                           type="checkbox" 
                           id="primary-${this._sanitizeId(field.name)}" 
                           value="${this._escapeHtml(field.name)}" 
                           onchange="pivotControlsModule.updateSelectedPrimary()">
                    <label class="form-check-label" for="primary-${this._sanitizeId(field.name)}">
                        ${this._escapeHtml(field.name)}
                    </label>
                </div>
            `;
            this.elements.primaryFieldCheckboxes.appendChild(li);
        });

        // Заполняем радиокнопки разбивки
        availableSplitFields.forEach(field => {
            const li = document.createElement('li');
            li.className = 'dropdown-item-text';
            li.innerHTML = `
                <div class="form-check">
                    <input class="form-check-input" 
                           type="radio" 
                           name="splitFieldSelection" 
                           id="split-${this._sanitizeId(field.name)}" 
                           value="${this._escapeHtml(field.name)}" 
                           onchange="pivotControlsModule.updateSelectedSplit()">
                    <label class="form-check-label" for="split-${this._sanitizeId(field.name)}">
                        ${this._escapeHtml(field.name)}
                    </label>
                </div>
            `;
            this.elements.splitFieldRadioButtons.appendChild(li);
        });

        // Выбираем первое поле по умолчанию
        if (availablePrimaryFields.length > 0) {
            const firstCheckbox = document.getElementById(`primary-${this._sanitizeId(availablePrimaryFields[0].name)}`);
            if (firstCheckbox) {
                firstCheckbox.checked = true;
                this.updateSelectedPrimary();
            }
        }
    }

    /**
     * Обновление выбранных основных полей (метрики/срезы)
     */
    updateSelectedPrimary() {
        const checkboxes = this.elements.primaryFieldCheckboxes.querySelectorAll('input[type="checkbox"]');
        const selected = [];

        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selected.push(checkbox.value);
            }
        });

        if (this.state.currentMode === 'time-series') {
            this.state.selectedMetrics = selected;
        } else {
            this.state.selectedSlices = selected;
        }

        // Обновляем текст дропбокса
        if (selected.length === 0) {
            this.elements.primaryFieldDropdownText.textContent = 
                this.state.currentMode === 'time-series' ? 'Выберите метрики' : 'Выберите срезы';
        } else if (selected.length === 1) {
            this.elements.primaryFieldDropdownText.textContent = selected[0];
        } else {
            this.elements.primaryFieldDropdownText.textContent = `Выбрано: ${selected.length}`;
        }

        console.log('📊 Выбрано полей:', selected);

        // Вызываем callback
        if (this.onStateChange) {
            this.onStateChange(this.state);
        }
    }

    /**
     * Обновление выбранного поля для разбивки
     */
    updateSelectedSplit() {
        const selectedRadio = document.querySelector('input[name="splitFieldSelection"]:checked');
        const selectedField = selectedRadio ? selectedRadio.value : '';

        this.state.selectedSplitField = selectedField;

        // Обновляем текст дропбокса
        if (selectedField) {
            this.elements.splitFieldDropdownText.textContent = `Разбивка: ${selectedField}`;
        } else {
            this.elements.splitFieldDropdownText.textContent = 
                this.state.currentMode === 'time-series' ? 'Только временные ряды' : 'Только срезы';
        }

        console.log('📊 Выбрана разбивка:', selectedField);

        // Вызываем callback
        if (this.onStateChange) {
            this.onStateChange(this.state);
        }
    }

    /**
     * Переключение режима (временные ряды / срезы)
     */
    switchMode(mode) {
        console.log('🔄 Переключение режима:', this.state.currentMode, '→', mode);

        this.state.currentMode = mode;

        // Обновляем активную кнопку
        this.elements.timeSeriesModeBtn.classList.toggle('active', mode === 'time-series');
        this.elements.slicesModeBtn.classList.toggle('active', mode === 'slices');

        // Обновляем заголовок
        if (mode === 'time-series') {
            this.elements.pivotModeTitle.innerHTML = '<i class="fas fa-clock me-2"></i>Режим отображения: Временные ряды';
        } else {
            this.elements.pivotModeTitle.innerHTML = '<i class="fas fa-layer-group me-2"></i>Режим отображения: Срезы';
        }

        // Сбрасываем выбранное поле разбивки
        this.state.selectedSplitField = '';
        const noSplitRadio = document.getElementById('noSplitFieldSelection');
        if (noSplitRadio) {
            noSplitRadio.checked = true;
        }

        // Обновляем селекты
        this.populateSelects();

        // Вызываем callback
        if (this.onStateChange) {
            this.onStateChange(this.state);
        }
    }

    /**
     * Переключение "Выбрать все"
     */
    toggleAllPrimary(checkbox) {
        const checkboxes = this.elements.primaryFieldCheckboxes.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = checkbox.checked);
        this.updateSelectedPrimary();
    }

    /**
     * Получение текущего состояния
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Установка состояния
     */
    setState(state) {
        if (!state) return;

        if (state.currentMode) {
            this.switchMode(state.currentMode);
        }

        // Восстанавливаем выбранные поля
        if (state.selectedMetrics || state.selectedSlices) {
            const selected = state.currentMode === 'time-series' ? 
                state.selectedMetrics : state.selectedSlices;

            const checkboxes = this.elements.primaryFieldCheckboxes.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = selected.includes(cb.value);
            });
            this.updateSelectedPrimary();
        }

        // Восстанавливаем разбивку
        if (state.selectedSplitField) {
            const radio = document.getElementById(`split-${this._sanitizeId(state.selectedSplitField)}`);
            if (radio) {
                radio.checked = true;
                this.updateSelectedSplit();
            }
        }
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
window.pivotControlsModule = new PivotControlsModule();
console.log('✅ Модуль pivot_controls.js загружен');

