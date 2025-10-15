/**
 * Модуль для загрузки и рендеринга данных сводной таблицы
 * Версия: 1.0.0
 */

class PivotLoaderModule {
    constructor() {
        this.rawData = null;
        this.dataMapping = null;
        this.containerElement = null;
    }

    /**
     * Инициализация модуля
     * @param {string} containerId - ID контейнера для таблицы
     */
    init(containerId) {
        this.containerElement = document.getElementById(containerId);
        
        if (!this.containerElement) {
            console.error('❌ Контейнер для таблицы не найден:', containerId);
            return false;
        }
        
        console.log('✅ Модуль загрузки инициализирован');
        return true;
    }

    /**
     * Установка данных
     * @param {Array} data - Массив данных
     * @param {Object} mapping - Конфигурация маппинга
     */
    setData(data, mapping) {
        this.rawData = data;
        this.dataMapping = mapping;
        
        console.log('📊 Данные установлены:', data.length, 'строк');
    }

    /**
     * Рендеринг сводной таблицы
     * @param {Object} options - Опции рендеринга
     */
    render(options = {}) {
        const {
            data = this.rawData,
            mapping = this.dataMapping,
            mode = 'time-series',
            splitField = ''
        } = options;

        if (!data || !mapping) {
            console.error('❌ Нет данных или маппинга для рендеринга');
            return;
        }

        console.log('🎨 Рендеринг сводной таблицы:', {
            rows: data.length,
            mode: mode,
            splitField: splitField
        });

        // Проверяем наличие функции renderNewPivotTable
        if (typeof renderNewPivotTable !== 'function') {
            console.error('❌ Функция renderNewPivotTable не найдена! Проверьте загрузку pivot_table_system.js');
            return;
        }

        try {
            // Вызываем функцию рендеринга из pivot_table_system.js
            renderNewPivotTable(
                data,
                mapping,
                mode,
                splitField
            );
            
            console.log('✅ Сводная таблица отрендерена');
        } catch (error) {
            console.error('❌ Ошибка рендеринга:', error);
        }
    }

    /**
     * Получение данных
     */
    getData() {
        return this.rawData;
    }

    /**
     * Получение маппинга
     */
    getMapping() {
        return this.dataMapping;
    }
}

// Создаем глобальный экземпляр
window.pivotLoaderModule = new PivotLoaderModule();
console.log('✅ Модуль pivot_loader.js загружен');

