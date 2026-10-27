# Рекомендации музыкальных товаров

Золотарёв Кирилл Алексеевич

## Датасет

Выбран **Amazon Review Data (2018), Musical Instruments, 5-core**.
[Первоисточник](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

## Получение данных

Скачайте с первоисточника 5-core для Musical Instruments и metadata той же категории.
Опубликованные ссылки:

- [Отзывы Musical_Instruments_5.json.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz)
- [Метаданные meta_Musical_Instruments.json.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz)

## Запуск

Из корня этого репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab experiments.ipynb
```