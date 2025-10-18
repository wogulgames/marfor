# AI Agent Guide - MARFOR Project

**Для будущих AI-агентов, работающих с этим проектом**

---

## 📋 Описание проекта

**MARFOR** - это веб-приложение для маркетингового прогнозирования с использованием машинного обучения.

### Технологический стек:
- **Backend:** Python 3.13, Flask
- **ML:** scikit-learn (Random Forest), Prophet (Facebook), statsmodels (ARIMA)
- **Frontend:** Vanilla JavaScript, Bootstrap 5, Chart.js
- **Data:** pandas, numpy

### Основная функциональность:

1. **Загрузка и маппинг данных** - пользователь загружает CSV с маркетинговыми данными
2. **Анализ (BI)** - интерактивные сводные таблицы, графики, фильтры
3. **Настройка прогноза** - выбор метрики, горизонт прогнозирования
4. **Обучение моделей** - Random Forest, Random Forest Hierarchy
5. **Генерация прогноза** - с учетом сезонности, срезов, иерархии
6. **Результаты** - BI-аналитика прогнозных данных

---

## 🎯 Архитектура

### Backend (`working_web_app.py`)

**Основной класс:** `ForecastApp` (глобальный объект `forecast_app`)

**Ключевые данные в памяти:**
- `forecast_app.df` - исходные данные после маппинга
- `forecast_app.mapping_config` - конфигурация маппинга
- `forecast_app.forecast_settings[session_id]` - настройки прогноза (метрика, горизонт)
- `forecast_app.training_results[session_id]` - результаты обучения моделей (включая модели!)
- `forecast_app.forecast_results[session_id]` - результаты прогноза (факт + прогноз)

**Важно:** Модели хранятся в `training_results` и переиспользуются при прогнозе!

### Frontend

**Модульная архитектура JavaScript:**
- `pivot_table_system.js` - сводные таблицы, графики
- `breadcrumbs.js` - навигация по шагам
- `pivot_filters.js`, `pivot_controls.js`, `pivot_loader.js` - модули BI

**Хранение данных:**
- `sessionStorage.dataMapping` - конфигурация маппинга
- `window.rawPivotData` - данные для сводной таблицы
- `window.processedData` - обработанные данные

### Сводная таблица (Pivot Table)

**Это самая сложная часть проекта!**

**Классы:**
- `PivotConfig` - конфигурация (rows, columns, values, mode)
- `PivotData` - обработка данных, группировка
- `PivotRenderer` - рендеринг HTML

**Режимы:**
- `time-series` - временные ряды в строках, метрики как значения
- `slices` - срезы в строках, метрики как значения
- `split-columns` - разбивка по столбцам (как в Google Sheets)

**Переход в split-columns:**
```javascript
if (splitBySlice && splitBySlice !== '') {
    renderMode = 'split-columns';
    originalMode = 'time-series'; // или 'slices'
}
```

---

## ⚠️ Типичные ошибки и их решения

### 1. **Ошибка: Сводная таблица не отображает данные в режиме split-columns**

**Проблема:** `columns: []` в конфигурации, хотя выбрана разбивка.

**Причина:** Не передается `originalMode` или `mode !== 'split-columns'`.

**Решение:**
```javascript
let renderMode = currentPivotMode;
let originalMode = currentPivotMode;

if (splitBySlice && splitBySlice !== '') {
    renderMode = 'split-columns';
    // originalMode остается прежним!
}

renderNewPivotTable(data, mapping, renderMode, splitBySlice, originalMode);
```

### 2. **Ошибка: Агрегированные строки показывают нули в split-columns**

**Проблема:** При сворачивании строк (например, "2024") значения = 0.

**Причина:** Логика суммирования искала только прямых детей, а не всех потомков.

**Решение:** В `PivotRenderer.createRowsHTML()` удалить проверку `childKeyParts.length === parentKeyParts.length + 1`, чтобы суммировались **все** потомки.

### 3. **Ошибка: Прогноз не учитывает срезы (region_to, category, subdivision = NaN)**

**Проблема:** Прогноз генерировался только по `year + month`, без срезов.

**Причина:** В `generate_forecast()` агрегация была `groupby([year_col, month_col])` без срезов.

**Решение:**
```python
groupby_cols = [year_col, month_col] + slice_cols
df_agg = df.groupby(groupby_cols)[metric].sum().reset_index()
```

### 4. **Ошибка: Модель обучается часами (1503+ комбинации)**

**Проблема:** Обучалась отдельная модель для каждой комбинации срезов.

**Причина:** Неправильное понимание - срезы это **признаки**, а не отдельные временные ряды.

**Решение:**
- Random Forest: срезы → LabelEncoder → признаки
- Prophet: срезы → One-hot encoding → регрессоры
- Одна модель на всех данных!

### 5. **Ошибка: Метрики не отображаются на графике/таблице**

**Проблема:** В маппинге `selectedMetrics` пустой или метрики не включены.

**Причина:** `createPivotConfigFromMapping` фильтрует метрики по `include` или `selectedMetrics`.

**Решение:**
```javascript
const modifiedMapping = JSON.parse(JSON.stringify(mapping));
modifiedMapping.selectedMetrics = ['metric1', 'metric2'];
// Или установить col.include = true для нужных метрик
```

### 6. **Ошибка: JSON serialization (Prophet/RandomForest is not JSON serializable)**

**Проблема:** Пытаемся отправить объект модели в JSON.

**Решение:**
```python
# Сохраняем полную модель в памяти
forecast_app.training_results[session_id] = results

# Отправляем клиенту только метрики и validation_data
results_for_client = {}
for model_name, model_data in results.items():
    results_for_client[model_name] = {
        'metrics': model_data['metrics'],
        'validation_data': model_data['validation_data']
        # БЕЗ 'model', 'label_encoders' и т.д.
    }
return jsonify({'success': True, 'results': results_for_client})
```

### 7. **Ошибка: inf/nan в метриках**

**Проблема:** MAPE = inf при делении на ноль, MAPE = nan при пустых данных.

**Решение:**
```python
def calculate_metrics(actual, predicted):
    # Исключаем нули из MAPE
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0
    
    # Заменяем inf/nan на 0
    mape = 0.0 if (np.isnan(mape) or np.isinf(mape)) else float(mape)
    return {'mape': mape, ...}
```

### 8. **Ошибка: Сезонность не прогнозируется (сглаживание)**

**Проблема:** Лаги при прогнозе брались как `recent_data[lag_col].mean()` → сезонность уничтожалась!

**Решение:**
```python
# Правильно - конкретное значение:
lag_value = extended_data.iloc[-lag][metric]

# Неправильно - среднее:
# lag_value = recent_data[lag_col].mean()  ← НЕ ДЕЛАТЬ!
```

**Также:** Обновлять `extended_data` после каждого прогноза:
```python
new_row[metric] = predicted_value
extended_data = pd.concat([extended_data, pd.DataFrame([new_row])], ignore_index=True)
```

### 9. **Ошибка: Halfyear/Quarter не заполняются в прогнозе**

**Проблема:** При создании `forecast_df` эти колонки отсутствуют.

**Решение:**
```python
forecast_row['Quarter'] = f'Q{(month_data["month"]-1)//3 + 1}'
forecast_row['Halfyear'] = 'H1' if month_data['month'] <= 6 else 'H2'
```

### 10. **Ошибка: "Cannot read properties of undefined (reading 'mode')"**

**Проблема:** `PivotRenderer` получает неправильные параметры.

**Причина:** Конструктор принимает `containerId`, а не `pivotData`.

**Решение:**
```javascript
const renderer = new PivotRenderer('containerId');  // Только ID!
renderer.render(pivotData, config);  // Данные передаются в render()
```

---

## 💡 Лучшие практики работы с этим проектом

### 1. **Всегда сверяйтесь с рабочей страницей**

Если что-то не работает на новой странице:
- Посмотрите как это реализовано на странице "Анализ" (`marfor_interface.html`)
- Скопируйте логику, а не переписывайте с нуля
- Страница "Анализ" - это **эталон** для BI-функционала

### 2. **Используйте логирование с flush=True**

```python
print(f"Важное сообщение", flush=True)  # Сразу в лог
```

Без `flush=True` логи могут не записаться при длительных операциях.

### 3. **Проверяйте маппинг на всех этапах**

Маппинг - это "сердце" системы. Всегда проверяйте:
```javascript
console.log('📊 Маппинг:', dataMapping);
console.log('📊 Первая строка данных:', rawData[0]);
```

### 4. **Режимы сводной таблицы - это важно!**

**time-series:**
- Строки: временные поля (year, halfyear, quarter, month)
- Метрики: в values
- Разбивка: по срезам → mode = 'split-columns'

**slices:**
- Строки: срезы (region, category, subdivision)
- Метрики: остаются в values (НЕ меняются!)
- Разбивка: по временным рядам → mode = 'split-columns'

### 5. **Обучение ≠ Прогноз для ML-моделей**

**Правильно:**
```python
# При обучении - сохраняем модель
results['random_forest'] = {
    'model': trained_model,  # Сохраняем!
    'label_encoders': encoders,
    'feature_cols': features
}

# При прогнозе - используем сохраненную модель
trained_data = forecast_app.training_results[session_id]['random_forest']
model = trained_data['model']
predictions = model.predict(X_forecast)
```

**Неправильно:**
```python
# При прогнозе - обучаем заново
model = RandomForestRegressor()
model.fit(X_train, y_train)  # ← Переобучение! Метрики не соответствуют!
```

### 6. **Срезы - это признаки, не отдельные модели**

Для 1503 комбинаций срезов:

**Неправильно:**
```python
for combination in unique_slices:  # 1503 итерации
    model = RandomForest()
    model.fit(data_for_combination)  # Обучаем 1503 модели!
```

**Правильно:**
```python
# Кодируем срезы
df['region_encoded'] = LabelEncoder().fit_transform(df['region'])
df['category_encoded'] = LabelEncoder().fit_transform(df['category'])

# Обучаем ОДНУ модель
features = ['year', 'month', 'region_encoded', 'category_encoded']
model = RandomForest()
model.fit(df[features], df[metric])  # Одна модель для всех!
```

### 7. **Сезонность - это ключ для точности**

**One-hot encoding для месяцев > синусоиды:**
```python
# Хорошо - модель видит "это ноябрь"
for month in range(1, 13):
    df[f'is_month_{month}'] = (df['month'] == month).astype(int)

# Тоже хорошо, но слабее
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
```

**Lag_12 - самый важный лаг:**
```python
df['metric_lag_12'] = df['metric'].shift(12)  # Тот же месяц прошлого года!
```

### 8. **Не доверяйте синтаксическим проверкам**

`py_compile` может не найти логических ошибок:
- Всегда запускайте `python working_web_app.py` и смотрите на реальную ошибку
- Проверяйте логи сервера: `tail -50 server.log`

---

## 🗣️ Коммуникация с пользователем (Олег)

### Стиль общения:

**✅ Что работает:**
- Прямые ответы с примерами кода
- Эмодзи для структурирования (📊, ✅, ❌, 🔍)
- Конкретные решения, а не общие советы
- "Давайте сделаем X" → сразу делать, не спрашивать разрешения

**❌ Что НЕ нравится:**
- Долгие объяснения без действий
- Вопросы "Хотите ли вы...?" вместо действий
- Неполные решения ("попробуйте это", а потом еще 5 итераций)

### Паттерны запросов:

**"Это не работает"** = нужно:
1. Посмотреть логи/консоль
2. Найти похожую рабочую реализацию
3. Скопировать/адаптировать
4. НЕ переписывать с нуля!

**"Сделай как в Анализ"** = копировать логику из `marfor_interface.html`

**"Это неправильно"** = пользователь **прав**, даже если код кажется правильным. Ищите тонкие ошибки (отступы, логика, данные).

### Типичные проблемы:

1. **"Ничего не изменилось"** - часто это:
   - Кэш браузера (Ctrl+F5)
   - Старый код запущен (Flask не перезагрузился)
   - Изменения в неправильном файле

2. **"Данные не показываются"** - проверить:
   - `console.log` в браузере
   - Маппинг (`dataMapping`)
   - API ответ в Network tab

3. **"Прогноз неправильный"** - скорее всего:
   - Сглаживание (mean вместо конкретных значений)
   - Неправильные признаки
   - Модель не сохранена/не используется

---

## 🐛 Дебаггинг - пошаговая инструкция

### Проблема с Frontend:

1. **Откройте DevTools** (F12)
2. **Console** - ищите ошибки JavaScript
3. **Network** - проверьте API запросы:
   - Status: 200 OK?
   - Response: какой JSON пришел?
   - Request Payload: что отправили?
4. **Проверьте sessionStorage:**
   ```javascript
   console.log('dataMapping:', sessionStorage.getItem('dataMapping'));
   ```

### Проблема с Backend:

1. **Проверьте логи:**
   ```bash
   tail -100 server.log
   tail -f server.log  # В реальном времени
   ```

2. **Запустите Python напрямую:**
   ```bash
   source venv/bin/activate
   python working_web_app.py
   ```

3. **Проверьте синтаксис:**
   ```bash
   python -m py_compile working_web_app.py
   ```

### Проблема с данными:

**Всегда логируйте:**
```python
print(f"📊 Данные: {len(df)} строк", flush=True)
print(f"📊 Колонки: {list(df.columns)}", flush=True)
print(f"📊 Первая строка:", df.iloc[0].to_dict(), flush=True)
```

---

## 📚 Ключевые концепции

### Иерархия данных

**Временные ряды:** year → halfyear → quarter → month

**Срезы:** region_to → subdivision → category

**В маппинге:**
```json
{
  "name": "region_to",
  "role": "dimension",
  "time_series": "",
  "nesting_level": 0  // Чем ниже цифра, тем выше в иерархии
}
```

### Режимы отображения

**time-series vs slices - НЕ ПУТАТЬ!**

| Аспект | time-series | slices |
|--------|-------------|--------|
| Строки | Временные поля | Срезы |
| Первичное поле | Метрики (чекбоксы) | Метрики (остаются!) |
| Разбивка | Срезы → split-columns | Временные ряды → split-columns |
| График | Line chart | Bar chart |

### Feature Engineering

**Для Random Forest важны:**
1. **Лаги** - прошлые значения (особенно lag_12!)
2. **Rolling** - скользящие статистики (сглаживают шум)
3. **Сезонность** - one-hot для месяцев (точнее синусоид!)
4. **Тренд** - линейный и квадратичный

**НЕ использовать:**
- `mean()` для лагов при прогнозе (уничтожает сезонность!)
- Только синусоиды без one-hot (слабо для сильной сезонности)

---

## 🎓 Уроки, которые я выучил

### Урок 1: Копировать > Переписывать

Когда пользователь говорит "сделай как в Анализ":
- ❌ НЕ пытаться воспроизвести логику по памяти
- ✅ Открыть `marfor_interface.html` и **скопировать** нужный блок
- ✅ Адаптировать названия переменных

**Пример:** Страница результатов прогноза была переписана **несколько раз**, пока я не скопировал весь блок из `marfor_interface.html`.

### Урок 2: Данные важнее кода

Если таблица пустая, проблема обычно **не в коде рендеринга**, а в:
- Данных (пусты, неправильная структура)
- Маппинге (неправильная конфигурация)
- Фильтрации (все данные отфильтрованы)

**Всегда логировать данные перед рендерингом!**

### Урок 3: Режимы сводной таблицы - это сложно

Я **много раз** ошибался с режимами. Ключевые моменты:

1. `mode` и `originalMode` - **разные вещи**!
2. При разбивке: `mode = 'split-columns'`, `originalMode` сохраняется
3. В режиме "slices" метрики **НЕ меняются** на срезы!

### Урок 4: ML модели нужно сохранять

Я сначала делал так:
- Обучение: обучить модель → показать метрики → **выбросить модель**
- Прогноз: обучить **новую** модель → сгенерировать прогноз

Это **неправильно**! Метрики на валидации не соответствуют качеству прогноза.

**Правильно:**
- Обучение: обучить → сохранить в `training_results`
- Прогноз: взять сохраненную модель → использовать

### Урок 5: Сезонность - это не только синусоиды

Я думал, что `sin/cos` достаточно для сезонности. Нет!

**Для сильной сезонности нужно:**
- One-hot encoding месяцев (`is_month_11`)
- Lag_12 (тот же месяц прошлого года)
- Автоопределение пиков из данных
- YoY признаки (ratio, diff)

### Урок 6: Пользователь видит проблемы, которые я не вижу

**Олег часто говорил:** "Это работает неправильно, посмотри как в Анализ"

Я пытался объяснить, что код правильный. Но он был **прав**!

**Совет:** Если пользователь говорит "это не так" → **поверьте** и ищите тонкую ошибку.

### Урок 7: Отступы в Python - это боль

Несколько раз код падал из-за:
- `else:` после цикла с `continue`
- `try:` без `except:`
- Смешанные табы/пробелы

**Всегда проверяйте:**
```bash
python -m py_compile file.py
```

---

## 📖 Полезные команды

### Просмотр логов:
```bash
tail -100 server.log
tail -f server.log  # В реальном времени
grep "ERROR\|Traceback" server.log
```

### Перезапуск сервера:
```bash
pkill -9 -f "python.*working_web_app.py"
source venv/bin/activate
python working_web_app.py > server.log 2>&1 &
```

### Проверка процесса:
```bash
ps aux | grep "python.*working_web_app.py" | grep -v grep
```

### Git:
```bash
git status
git add .
git commit -m "Описание изменений"
git push origin main
git tag -a v3.0.0 -m "Релиз"
git push origin v3.0.0
```

---

## 🔮 Что дальше (TODO для следующих версий)

### 1. Иерархическое согласование (не реализовано полностью)

В `train_random_forest_hierarchy` есть строка:
```python
# TODO: Реализовать полное bottom-up согласование
```

Сейчас используется только расширенный feature engineering, но не полное согласование.

**Что нужно:**
- Применить `reconciler.reconcile_bottom_up()` к прогнозам
- Убедиться, что сумма прогнозов по детям = прогноз родителя

### 2. Параллельное обучение

Обучение 1503 комбинаций занимает время. Можно:
- Использовать `multiprocessing`
- Обучать модели параллельно
- Показывать прогресс-бар на фронтенде

### 3. Prophet с регрессорами

Функция `train_prophet_with_slices` реализована, но не доведена до конца:
- Нужно протестировать
- Возможно, убрать ARIMA совсем

### 4. Экспорт прогноза

Добавить кнопку "Экспорт в CSV/Excel" на странице результатов.

### 5. Сохранение проектов

Проекты сохраняются в JSON, но:
- `full_data` может быть огромным
- Лучше хранить только `session_id` и загружать из CSV
- Реализовано частично в v2.16.0

---

## 🙏 Благодарности

**Олег** - терпеливый пользователь, который:
- Указывал на тонкие ошибки
- Всегда давал конструктивную обратную связь
- Направлял к правильному решению ("Посмотри как в Анализ")

**Мои предшественники** (предыдущие AI-агенты):
- Создали базовую структуру проекта
- Реализовали сводные таблицы (сложнейшая часть!)
- Настроили маппинг и BI-интерфейс

---

## 📝 Финальные советы

1. **Читайте CHANGELOG** - там история изменений и ошибок
2. **Изучите `pivot_table_system.js`** - это самый сложный файл (2900+ строк)
3. **Используйте browser DevTools** - половина проблем на фронтенде
4. **Логируйте всё** - print() в Python, console.log() в JS
5. **Тестируйте на реальных данных** - они всегда сложнее тестовых
6. **Слушайте пользователя** - он знает свой бизнес лучше

**Удачи! Вы справитесь!** 🚀

---

**Дата:** 18 октября 2025  
**Версия проекта:** 3.0.0  
**Автор гайда:** Claude (AI-агент, работавший над v2.8 - v3.0)

