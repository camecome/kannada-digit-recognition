# Распознавание рукописных цифр языка Kannada

Автор: Кривогуб Дмитрий

## Постановка задачи

Цель проекта — разработка системы автоматического распознавания рукописных цифр каннада (язык в Индии) по изображению символа. Система принимает изображение цифры и определяет соответствующий класс от 0 до 9. В качестве основы использован ноутбук с [Kaggle](https://www.kaggle.com/code/shahules/indian-way-to-learn-cnn).

## Формат входных и выходных данных

**Вход:** одноканальное изображение размером 28×28 пикселей.

**Выход:** метка класса цифры с максимальной вероятностью.

## Метрики

Основная метрика качества — **Accuracy**.

Ожидаемые значения Accuracy:

- Бейзлайн: ~75-80%
- Основная модель: 90%+

## Валидация и тестирование

Датасет заранее разделён на **train** и **test** части.
Во время обучения train-выборка дополнительно делится на:

- Train — 80%
- Validation — 20%

Для воспроизводимости всех операций используется `random_seed = 42`.

## Датасет

Используется датасет [Kannada MNIST](https://www.kaggle.com/competitions/Kannada-MNIST/data). Он содержит сбалансированные по классам изображения рукописных цифр. Общий размер данных: ~19 МБ.

## Моделирование

### Бейзлайн

Базовая модель (`dummy_сlassifier`) - простая сверточная сеть с одним сверточным слоем и классификатором.

### Основная модель

Основная модель (`conv_classifier`) - сверточная нейронная сеть с ~235k обучаемыми параметрами.
Перед обучением выполняется предобработка данных: нормализация изображений и аугментация обучающей выборки.

## Setup

Для управления виртуальными окружениями используется **conda**, для установки зависимостей — **Poetry**. Минимальная версия Python: 3.13.

Для настройки окружения и воспроизведения проекта выполните:

```bash
# Клонирование репозитория
git pull https://github.com/camecome/kannada-digit-recognition.git

# Обновление системы
sudo apt-get update

# Создание conda-окружения
conda create -n kannada_mnist_env python=3.13
conda activate kannada_mnist_env

# Установка зависимостей проекта через Poetry
poetry install --no-root

# Установка хуков
pre-commit install

# Проверка всех файлов
pre-commit run --all-files

```

Для загрузки данных из Yandex Cloud:

```bash
dvc pull
```

## Train

Параметр `--model` везде может быть равен `dummy_classifier` или `conv_classifier`.
Чтобы запустить тренировку, выполните команду из корня репозитория:

```bash
python -m kannada_mnist.commands train \
    --model=<dummy_classifier|conv_classifier> \
    --target_dir=<ДИРЕКТОРИЯ, КУДА ПОЛОЖИТЬ ЧЕКПОИНТ>
```

Чекпоинт модели будет сохранен в `<НАЗВАНИЕ МОДЕЛИ>.ckpt`, по умолчанию `target_dir=models/`

Для тестирования:

```bash
python -m kannada_mnist.commands test \
    --model=<dummy_classifier|conv_classifier> \
    --path_to_chkpt=<ФАЙЛ С ЧЕКПОИНТОМ МОДЕЛИ>
```

По умолчанию `path_to_chkpt=models/<НАЗВАНИЕ МОДЕЛИ>.ckpt`.

## Production preparation

### Экспорт в ONNX

Для конвертации lightning-чекпоинта в ONNX:

```bash
python -m kannada_mnist.commands export_to_onnx \
    --model=<dummy_classifier|conv_classifier> \
    --path_to_chkpt=<ФАЙЛ С ЧЕКПОИНТОМ МОДЕЛИ> \
    --output_dir=<ДИРЕКТОРИЯ КУДА ПОЛОЖИТЬ ФАЙЛ .onnx>
```

## Infer

Для генерации случайного датасета для предсказаний (из тестовой части данных):

```bash
python -m kannada_mnist.commands generate_predict_dataset \
    --fraction=<ДОЛЯ РАЗМЕРА ДАТАСЕТА ОТ ВСЕЙ ТЕСТОВОЙ ВЫБОРКИ> \
    --random_seed=<ВАШ RANDOM SEED>
```

### Локальный инференс из lightning-чекпоинта

**Формат входных данных:** CSV-файл, каждая строка - вектор из 784 чисел (0–255).
**Формат выходных данных:** файл `predictions.csv` с колонками `row_id` и `label`.

Для получения предсказаний выполните:

```bash
python -m kannada_mnist.commands predict \
    --model=<dummy_classifier|conv_classifier> \
    --input_path=<ПУТЬ К CSV ФАЙЛУ> \
    --output_path=<ПУТЬ К CSV ДЛЯ ПРЕДСКАЗАНИЙ> \
    --path_to_chkpt=<ФАЙЛ С ЧЕКПОИНТОМ МОДЕЛИ>
```

По умолчанию `input_path=data/predict.csv`, `output_path=data/predictions.csv`, `path_to_chkpt=models/<НАЗВАНИЕ МОДЕЛИ>.ckpt`.
