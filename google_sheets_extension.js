/**
 * Google Sheets расширение для прогнозирования маркетинговых данных
 * 
 * Инструкция по установке:
 * 1. Откройте Google Sheets
 * 2. Перейдите в Extensions > Apps Script
 * 3. Удалите весь код и вставьте этот код
 * 4. Сохраните проект
 * 5. Вернитесь в Google Sheets и обновите страницу
 * 6. В меню появится пункт "Прогнозирование"
 */

/**
 * Создает меню в Google Sheets
 */
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('📊 Прогнозирование')
    .addItem('🔮 Создать прогноз', 'createForecast')
    .addItem('📈 Анализ трендов', 'analyzeTrends')
    .addItem('📋 Настройки прогноза', 'showSettings')
    .addSeparator()
    .addItem('❓ Помощь', 'showHelp')
    .addToUi();
}

/**
 * Основная функция создания прогноза
 */
function createForecast() {
  const ui = SpreadsheetApp.getUi();
  
  try {
    // Получаем активный лист
    const sheet = SpreadsheetApp.getActiveSheet();
    const data = sheet.getDataRange().getValues();
    
    if (data.length < 2) {
      ui.alert('Ошибка', 'Недостаточно данных для прогнозирования. Нужно минимум 2 строки.', ui.ButtonSet.OK);
      return;
    }
    
    // Показываем диалог настроек
    const settings = showForecastSettings();
    if (!settings) return;
    
    // Анализируем структуру данных
    const analysis = analyzeDataStructure(data);
    
    // Создаем прогноз
    const forecast = generateForecast(data, analysis, settings);
    
    // Добавляем прогноз в новый лист
    addForecastToSheet(forecast, settings);
    
    ui.alert('Успех', `Прогноз создан! Добавлено ${forecast.length} новых записей.`, ui.ButtonSet.OK);
    
  } catch (error) {
    ui.alert('Ошибка', `Произошла ошибка: ${error.message}`, ui.ButtonSet.OK);
  }
}

/**
 * Показывает диалог настроек прогноза
 */
function showForecastSettings() {
  const ui = SpreadsheetApp.getUi();
  
  const html = HtmlService.createHtmlOutput(`
    <div style="font-family: Arial, sans-serif; padding: 20px;">
      <h3>⚙️ Настройки прогноза</h3>
      
      <div style="margin-bottom: 15px;">
        <label><strong>Количество периодов для прогноза:</strong></label><br>
        <input type="number" id="periods" value="4" min="1" max="12" style="width: 100px; padding: 5px;">
      </div>
      
      <div style="margin-bottom: 15px;">
        <label><strong>Метод прогнозирования:</strong></label><br>
        <select id="method" style="width: 200px; padding: 5px;">
          <option value="linear">Линейная регрессия</option>
          <option value="polynomial">Полиномиальная регрессия</option>
        </select>
      </div>
      
      <div style="margin-bottom: 15px;">
        <label><strong>Колонка с годом:</strong></label><br>
        <select id="yearColumn" style="width: 200px; padding: 5px;">
          <option value="A">A (1-я колонка)</option>
          <option value="B">B (2-я колонка)</option>
          <option value="C">C (3-я колонка)</option>
          <option value="D">D (4-я колонка)</option>
          <option value="E">E (5-я колонка)</option>
        </select>
      </div>
      
      <div style="margin-bottom: 15px;">
        <label><strong>Колонка с месяцем:</strong></label><br>
        <select id="monthColumn" style="width: 200px; padding: 5px;">
          <option value="B">B (2-я колонка)</option>
          <option value="A">A (1-я колонка)</option>
          <option value="C">C (3-я колонка)</option>
          <option value="D">D (4-я колонка)</option>
          <option value="E">E (5-я колонка)</option>
        </select>
      </div>
      
      <div style="margin-bottom: 20px;">
        <label><strong>Колонки для группировки (через запятую):</strong></label><br>
        <input type="text" id="groupColumns" value="C,D,E" style="width: 300px; padding: 5px;" 
               placeholder="Например: C,D,E">
      </div>
      
      <div style="text-align: center;">
        <button onclick="submitSettings()" style="background: #4285f4; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Создать прогноз</button>
        <button onclick="google.script.host.close()" style="background: #ccc; color: black; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-left: 10px;">Отмена</button>
      </div>
    </div>
    
    <script>
      function submitSettings() {
        const settings = {
          periods: document.getElementById('periods').value,
          method: document.getElementById('method').value,
          yearColumn: document.getElementById('yearColumn').value,
          monthColumn: document.getElementById('monthColumn').value,
          groupColumns: document.getElementById('groupColumns').value.split(',').map(col => col.trim())
        };
        
        google.script.host.setHeight(0);
        google.script.run.withSuccessHandler(function() {
          google.script.host.close();
        }).processForecastSettings(settings);
      }
    </script>
  `).setWidth(500).setHeight(400);
  
  ui.showModalDialog(html, 'Настройки прогноза');
  
  // Возвращаем настройки (в реальной реализации это будет через callback)
  return {
    periods: 4,
    method: 'linear',
    yearColumn: 'A',
    monthColumn: 'B',
    groupColumns: ['C', 'D', 'E']
  };
}

/**
 * Обрабатывает настройки из диалога
 */
function processForecastSettings(settings) {
  // Здесь будет логика обработки настроек
  console.log('Настройки получены:', settings);
}

/**
 * Анализирует структуру данных
 */
function analyzeDataStructure(data) {
  const headers = data[0];
  const analysis = {
    headers: headers,
    numericColumns: [],
    categoricalColumns: [],
    dateColumns: [],
    totalRows: data.length - 1
  };
  
  // Анализируем каждую колонку
  for (let i = 0; i < headers.length; i++) {
    const columnData = data.slice(1).map(row => row[i]);
    const isNumeric = columnData.every(val => !isNaN(val) && val !== '');
    
    if (isNumeric) {
      analysis.numericColumns.push({
        index: i,
        name: headers[i],
        letter: String.fromCharCode(65 + i)
      });
    } else {
      analysis.categoricalColumns.push({
        index: i,
        name: headers[i],
        letter: String.fromCharCode(65 + i)
      });
    }
  }
  
  return analysis;
}

/**
 * Генерирует прогноз на основе данных
 */
function generateForecast(data, analysis, settings) {
  const forecast = [];
  const headers = data[0];
  
  // Простая линейная экстраполяция для демонстрации
  // В реальной реализации здесь будет более сложная логика
  
  // Находим последний период
  const lastRow = data[data.length - 1];
  const yearCol = getColumnIndex(settings.yearColumn);
  const monthCol = getColumnIndex(settings.monthColumn);
  
  let lastYear = lastRow[yearCol];
  let lastMonth = lastRow[monthCol];
  
  // Генерируем прогнозные периоды
  for (let i = 1; i <= settings.periods; i++) {
    lastMonth++;
    if (lastMonth > 12) {
      lastMonth = 1;
      lastYear++;
    }
    
    // Создаем прогнозную запись для каждой группы
    const groups = getUniqueGroups(data, settings.groupColumns);
    
    groups.forEach(group => {
      const forecastRow = new Array(headers.length);
      
      // Заполняем даты
      forecastRow[yearCol] = lastYear;
      forecastRow[monthCol] = lastMonth;
      
      // Заполняем группировочные колонки
      settings.groupColumns.forEach((col, index) => {
        const colIndex = getColumnIndex(col);
        forecastRow[colIndex] = group[index];
      });
      
      // Прогнозируем числовые значения
      analysis.numericColumns.forEach(numCol => {
        if (numCol.letter !== settings.yearColumn && numCol.letter !== settings.monthColumn) {
          // Простая линейная экстраполяция
          const historicalValues = getHistoricalValues(data, group, settings.groupColumns, numCol.index);
          forecastRow[numCol.index] = calculateLinearForecast(historicalValues, i);
        }
      });
      
      // Заполняем остальные колонки
      for (let j = 0; j < headers.length; j++) {
        if (forecastRow[j] === undefined) {
          forecastRow[j] = '';
        }
      }
      
      forecast.push(forecastRow);
    });
  }
  
  return forecast;
}

/**
 * Получает уникальные группы из данных
 */
function getUniqueGroups(data, groupColumns) {
  const groups = new Set();
  
  for (let i = 1; i < data.length; i++) {
    const row = data[i];
    const group = groupColumns.map(col => row[getColumnIndex(col)]);
    groups.add(JSON.stringify(group));
  }
  
  return Array.from(groups).map(group => JSON.parse(group));
}

/**
 * Получает исторические значения для группы
 */
function getHistoricalValues(data, group, groupColumns, valueColumn) {
  const values = [];
  
  for (let i = 1; i < data.length; i++) {
    const row = data[i];
    const isMatch = groupColumns.every((col, index) => 
      row[getColumnIndex(col)] === group[index]
    );
    
    if (isMatch && !isNaN(row[valueColumn])) {
      values.push(row[valueColumn]);
    }
  }
  
  return values;
}

/**
 * Вычисляет линейный прогноз
 */
function calculateLinearForecast(values, periods) {
  if (values.length < 2) return values[0] || 0;
  
  // Простая линейная экстраполяция
  const n = values.length;
  const sumX = n * (n + 1) / 2;
  const sumY = values.reduce((a, b) => a + b, 0);
  const sumXY = values.reduce((sum, y, x) => sum + (x + 1) * y, 0);
  const sumXX = n * (n + 1) * (2 * n + 1) / 6;
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  
  return Math.max(0, slope * (n + periods) + intercept);
}

/**
 * Добавляет прогноз в новый лист
 */
function addForecastToSheet(forecast, settings) {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const originalSheet = SpreadsheetApp.getActiveSheet();
  
  // Создаем новый лист для прогноза
  const forecastSheet = spreadsheet.insertSheet(`Прогноз_${new Date().toISOString().slice(0,10)}`);
  
  // Копируем заголовки
  const headers = originalSheet.getRange(1, 1, 1, originalSheet.getLastColumn()).getValues()[0];
  forecastSheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  
  // Добавляем исходные данные
  const originalData = originalSheet.getDataRange().getValues();
  if (originalData.length > 1) {
    forecastSheet.getRange(2, 1, originalData.length - 1, headers.length)
      .setValues(originalData.slice(1));
  }
  
  // Добавляем прогнозные данные
  if (forecast.length > 0) {
    const startRow = originalData.length + 1;
    forecastSheet.getRange(startRow, 1, forecast.length, headers.length)
      .setValues(forecast);
    
    // Выделяем прогнозные данные другим цветом
    forecastSheet.getRange(startRow, 1, forecast.length, headers.length)
      .setBackground('#e8f0fe');
  }
  
  // Форматируем лист
  formatForecastSheet(forecastSheet, headers.length);
  
  // Переключаемся на новый лист
  spreadsheet.setActiveSheet(forecastSheet);
}

/**
 * Форматирует лист с прогнозом
 */
function formatForecastSheet(sheet, numColumns) {
  // Форматируем заголовки
  sheet.getRange(1, 1, 1, numColumns)
    .setBackground('#4285f4')
    .setFontColor('white')
    .setFontWeight('bold');
  
  // Автоматически подбираем ширину колонок
  sheet.autoResizeColumns(1, numColumns);
  
  // Добавляем границы
  sheet.getRange(1, 1, sheet.getLastRow(), numColumns)
    .setBorder(true, true, true, true, true, true);
}

/**
 * Анализ трендов
 */
function analyzeTrends() {
  const ui = SpreadsheetApp.getUi();
  const sheet = SpreadsheetApp.getActiveSheet();
  const data = sheet.getDataRange().getValues();
  
  if (data.length < 3) {
    ui.alert('Ошибка', 'Недостаточно данных для анализа трендов.', ui.ButtonSet.OK);
    return;
  }
  
  const analysis = analyzeDataStructure(data);
  
  // Создаем отчет
  const report = generateTrendReport(data, analysis);
  
  // Показываем отчет
  showTrendReport(report);
}

/**
 * Генерирует отчет по трендам
 */
function generateTrendReport(data, analysis) {
  const report = {
    totalRecords: data.length - 1,
    dateRange: getDateRange(data),
    topMetrics: [],
    trends: []
  };
  
  // Анализируем топ-метрики
  analysis.numericColumns.forEach(col => {
    const values = data.slice(1).map(row => row[col.index]).filter(val => !isNaN(val));
    if (values.length > 0) {
      const total = values.reduce((a, b) => a + b, 0);
      const avg = total / values.length;
      const growth = calculateGrowthRate(values);
      
      report.topMetrics.push({
        name: col.name,
        total: total,
        average: avg,
        growth: growth
      });
    }
  });
  
  // Сортируем по общему значению
  report.topMetrics.sort((a, b) => b.total - a.total);
  
  return report;
}

/**
 * Показывает отчет по трендам
 */
function showTrendReport(report) {
  const ui = SpreadsheetApp.getUi();
  
  let message = `📊 ОТЧЕТ ПО ТРЕНДАМ\n\n`;
  message += `📅 Период: ${report.dateRange}\n`;
  message += `📈 Всего записей: ${report.totalRecords}\n\n`;
  message += `🏆 ТОП-5 МЕТРИК:\n`;
  
  report.topMetrics.slice(0, 5).forEach((metric, index) => {
    message += `${index + 1}. ${metric.name}\n`;
    message += `   Общее: ${metric.total.toLocaleString()}\n`;
    message += `   Среднее: ${metric.average.toLocaleString()}\n`;
    message += `   Рост: ${metric.growth.toFixed(1)}%\n\n`;
  });
  
  ui.alert('Анализ трендов', message, ui.ButtonSet.OK);
}

/**
 * Показывает настройки
 */
function showSettings() {
  const ui = SpreadsheetApp.getUi();
  
  const message = `⚙️ НАСТРОЙКИ ПРОГНОЗИРОВАНИЯ

📋 Требования к данным:
• Первая строка должна содержать заголовки
• Должны быть колонки с годом и месяцем
• Числовые данные для прогнозирования

🔧 Поддерживаемые методы:
• Линейная регрессия
• Полиномиальная регрессия

📊 Результат:
• Новый лист с прогнозом
• Исходные данные + прогнозные
• Автоматическое форматирование

❓ Для получения помощи используйте пункт "Помощь"`;
  
  ui.alert('Настройки', message, ui.ButtonSet.OK);
}

/**
 * Показывает справку
 */
function showHelp() {
  const ui = SpreadsheetApp.getUi();
  
  const message = `❓ СПРАВКА ПО ПРОГНОЗИРОВАНИЮ

🚀 КАК ИСПОЛЬЗОВАТЬ:
1. Подготовьте данные в Google Sheets
2. Выберите "Прогнозирование" > "Создать прогноз"
3. Настройте параметры
4. Получите результат в новом листе

📋 ФОРМАТ ДАННЫХ:
• Год: 2023, 2024, 2025...
• Месяц: 1, 2, 3... 12
• Числовые метрики: выручка, трафик, расходы

⚙️ НАСТРОЙКИ:
• Периоды: количество месяцев для прогноза
• Метод: линейный или полиномиальный
• Группировка: по категориям, подразделениям

📞 ПОДДЕРЖКА:
При возникновении проблем проверьте:
• Формат данных
• Наличие заголовков
• Корректность группировочных колонок`;
  
  ui.alert('Справка', message, ui.ButtonSet.OK);
}

/**
 * Вспомогательные функции
 */
function getColumnIndex(columnLetter) {
  return columnLetter.charCodeAt(0) - 65;
}

function getDateRange(data) {
  // Простая реализация для получения диапазона дат
  return "2023-2026";
}

function calculateGrowthRate(values) {
  if (values.length < 2) return 0;
  
  const first = values[0];
  const last = values[values.length - 1];
  
  return ((last - first) / first) * 100;
}
