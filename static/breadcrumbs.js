/**
 * Модуль для управления хлебными крошками навигации
 * Версия: 1.0.0
 */

class BreadcrumbsModule {
    constructor() {
        this.steps = [
            {
                id: 1,
                name: 'Загрузка данных',
                shortName: 'Загрузка',
                url: '/',
                icon: '📁'
            },
            {
                id: 2,
                name: 'Маппинг данных',
                shortName: 'Маппинг',
                url: '/forecast/mapping',
                icon: '🔧'
            },
            {
                id: 3,
                name: 'Анализ данных',
                shortName: 'Анализ',
                url: '/forecast',
                icon: '📈'
            },
            {
                id: 4,
                name: 'Настройка прогноза',
                shortName: 'Настройка',
                url: '/forecast/settings',
                icon: '⚙️'
            },
            {
                id: 5,
                name: 'Обучение моделей',
                shortName: 'Обучение',
                url: '/forecast/training',
                icon: '🎓'
            },
            {
                id: 6,
                name: 'Результаты прогноза',
                shortName: 'Результаты',
                url: '/forecast/results',
                icon: '📊'
            }
        ];
    }

    /**
     * Рендеринг хлебных крошек
     * @param {number} currentStep - Текущий шаг (1-5)
     * @param {string} containerId - ID контейнера для вставки
     */
    render(currentStep, containerId = 'breadcrumbContainer') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn('Контейнер для хлебных крошек не найден:', containerId);
            return;
        }

        const html = `
            <div class="breadcrumb-container">
                <div class="step-indicator">
                    ${this.steps.map((step, index) => this._renderStep(step, currentStep, index)).join('')}
                </div>
            </div>
        `;

        container.innerHTML = html;
        console.log('✅ Хлебные крошки отрендерены, текущий шаг:', currentStep);
    }

    /**
     * Рендеринг отдельного шага
     * @private
     */
    _renderStep(step, currentStep, index) {
        let stepClass = 'step-item';
        
        if (step.id < currentStep) {
            stepClass += ' completed';
        } else if (step.id === currentStep) {
            stepClass += ' active';
        } else {
            stepClass += ' disabled';
        }

        const isClickable = step.id < currentStep;
        const onClick = isClickable ? `onclick="breadcrumbsModule.navigateTo(${step.id})"` : '';
        const cursor = isClickable ? 'cursor: pointer;' : '';

        return `
            <div class="${stepClass}">
                <div class="step-circle" ${onClick} style="${cursor}">
                    ${step.id < currentStep ? '' : step.id}
                </div>
                <div class="step-label">${step.shortName}</div>
                ${index < this.steps.length - 1 ? '<div class="step-connector"></div>' : ''}
            </div>
        `;
    }

    /**
     * Навигация к указанному шагу
     * @param {number} stepId - ID шага
     */
    navigateTo(stepId) {
        const step = this.steps.find(s => s.id === stepId);
        if (!step) {
            console.error('Шаг не найден:', stepId);
            return;
        }

        console.log('🔄 Переход к шагу:', step.name);

        // Добавляем session_id если он есть
        const currentSessionId = sessionStorage.getItem('currentSessionId') || 
                                 sessionStorage.getItem('sessionId');
        
        let url = step.url;
        if (currentSessionId && url !== '/') {
            url += `?session_id=${currentSessionId}`;
        }

        window.location.href = url;
    }

    /**
     * Получение текущего шага на основе URL
     * @returns {number} Номер шага (1-6)
     */
    getCurrentStepFromUrl() {
        const path = window.location.pathname;
        
        // Проверяем наличие обработанных данных для различения шагов 1 и 3
        const hasProcessedData = sessionStorage.getItem('dataMapping') !== null;
        
        if (path === '/' || path === '/forecast') {
            // Если данные обработаны (маппинг применен) - это шаг 3 (Анализ)
            // Если нет - это шаг 1 (Загрузка)
            return hasProcessedData ? 3 : 1;
        } else if (path.includes('/mapping')) {
            return 2;
        } else if (path.includes('/settings')) {
            return 4;
        } else if (path.includes('/training')) {
            return 5;
        } else if (path.includes('/results')) {
            return 6;
        }
        
        return 1;
    }

    /**
     * Автоматический рендеринг на основе текущего URL
     */
    autoRender(containerId = 'breadcrumbContainer') {
        const currentStep = this.getCurrentStepFromUrl();
        this.render(currentStep, containerId);
    }
}

// Создаем глобальный экземпляр
window.breadcrumbsModule = new BreadcrumbsModule();
console.log('✅ Модуль breadcrumbs.js загружен');

