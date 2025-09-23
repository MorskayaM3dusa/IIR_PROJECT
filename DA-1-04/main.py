import pandas as pd
import numpy as np

def is_numeric_dataframe(df):
    """
    Проверяет, что все столбцы в DataFrame имеют числовой тип данных
    Args:
        df (pd.DataFrame): DataFrame для проверки
    Returns:
        bool: True, если все столбцы числовые, False в противном случае
    """
    return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)


def load_data_from_file(filename):
    """
    Загружает DataFrame из CSV-файла и возвращает названия столбцов.
    Args:
        filename: Имя файла для загрузки.
    Returns:
        tuple: (pd.DataFrame, list) Загруженный DataFrame и список названий столбцов,
               или (None, None) в случае ошибки.
    """
    try:
        df = pd.read_csv(filename)
        if df is not None and not df.empty:
            column = df.columns[0]
            return df, column
        else:
            print(f"Загруженный DataFrame пуст.")
            return None, None
    except Exception as e:
        print(f"Не удалось загрузить данные из файла {filename}: {e}")
        return None, None


def generate_synthetic_data(categories_list, category_column, num_samples, seed):
    """Генерирует синтетический датасет."""
    np.random.seed(seed)
    categories = np.random.choice(categories_list, size=num_samples)
    df = pd.DataFrame({category_column: categories})
    return df


def process_dataframe(df, category_column):

    """
    Кодирует категориальные признаки и
    проверяет типы данных
    Args:
        df (pd.DataFrame or None): Исходный DataFrame
        category_column (str): Название столбца с категориальными данными
    Returns:
        tuple: Кортеж из (исходного DataFrame, закодированного DataFrame),
               или (None, None) в случае неудачи
    Raises:
        ValueError: Если входные параметры некорректны
    """

    if category_column not in df.columns:
        raise ValueError(f"DataFrame должен содержать столбец '{category_column}'")

    df_encoded = pd.get_dummies(df, columns=[category_column])
    df_encoded = df_encoded.astype(int)

    if not is_numeric_dataframe(df_encoded):
        raise ValueError("После кодирования не все типы данных числовые")

    return df, df_encoded




def save_dataframe_to_file(df, filename, index=False):
    """
    Сохраняет DataFrame в CSV-файл.

    Args:
        df (pd.DataFrame): DataFrame для сохранения.
        filename (str): Имя файла для сохранения.
        index (bool): Сохранять ли индекс строк
    """
    try:
        df.to_csv(filename, index=index)
        print(f"DataFrame успешно сохранен в файл {filename}.")
    except Exception as e:
        print(f"Не удалось сохранить DataFrame в файл {filename}: {e}")



def process_data(categories_list=None, category_column = 'category', num_samples=100, filename=None, save_filename='processed_data.csv', seed=123):
    """
    Главная функция для генерации, загрузки, обработки и сохранения датасета

    Args:
        categories_list (list): Список возможных значений для категориального признака
        category_column (str): Название столбца с категориальными данными
        num_samples (int): Количество строк в генерируемом датасете
        filename (str, optional): Имя файла для загрузки датасета
        save_filename (str, optional): Имя файла для сохранения закодированного датасета
        seed (int, optional): Зерно для генератора случайных чисел
    """

    df = None
    if filename:
        df, category_column = load_data_from_file(filename)
    else:
        if not categories_list:
            raise ValueError("categories_list не может быть пустым при генерации данных")
        if num_samples <= 0:
            raise ValueError("num_samples должно быть больше нуля при генерации данных")
        df = generate_synthetic_data(categories_list, category_column, num_samples, seed)

    try:
        original_data, encoded_data = process_dataframe(df, category_column)
    except ValueError as e:
        print(f"Генерация не удалась: {e}")
        original_data, encoded_data = None, None

    if original_data is not None and encoded_data is not None:
        if save_filename:
            save_dataframe_to_file(encoded_data, save_filename)
    else:
        print("Не удалось сгенерировать или загрузить и обработать данные.")
        

if __name__ == "__main__":
    process_data(categories_list=['red', 'blue', 'green'], category_column = 'test', num_samples=10, save_filename='encoded_synthetic_data.csv')

