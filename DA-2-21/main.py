import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

def load_data_from_file(filename):
    """
    Загружает DataFrame из CSV-файла и его
    Args:
        filename: Имя файла для загрузки.
    Returns:
        DataFrame: Загруженный DataFrame, или None в случае ошибки.
    """
    try:
        df = pd.read_csv(filename)
        if df is not None and not df.empty:
            return df
        else:
            print(f"Загруженный DataFrame пуст.")
            return None
    except Exception as e:
        print(f"Не удалось загрузить данные из файла {filename}: {e}")
        return None

def load_data(csv_file_path=None):
    """
    Загрузка данных: по умолчанию diabetes, либо CSV файл
    Args:
        csv_file_path (str): путь к CSV файлу (опционально)
    Returns:
        DataFrame: pandas DataFrame с данными
    """
    if csv_file_path is None:
        diabetes = load_diabetes()
        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        print("Используется встроенный датасет diabetes")
        return df
    else:
        df = load_data_from_file(csv_file_path)
        if df is None:
            raise ValueError(f"Не удалось загрузить данные из файла {csv_file_path}")
        
        print(f"CSV файл {csv_file_path} успешно загружен")
        print(f"Размер данных: {df.shape}")
        print(f"Колонки: {list(df.columns)}")
        return df

def select_numeric_feature(df, feature_name=None):
    """
    Выбор числового признака для биннинга
    Args:
        df (DataFrame): исходный DataFrame
        feature_name (str): название числового признака (если None - первый числовой)
    Returns:
        tuple: (Series признак, str название признака)
    """
    if feature_name is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("В DataFrame нет числовых колонок")
        feature_name = numeric_columns[0]
        print(f"Признак не указан, выбран первый числовой: {feature_name}")
    
    if feature_name not in df.columns:
        raise ValueError(f"Признак '{feature_name}' не найден в DataFrame")
    
    feature_series = df[feature_name]

    if not pd.api.types.is_numeric_dtype(feature_series):
        raise ValueError(f"Признак '{feature_name}' не является числовым")
    
    print(f"Выбран признак: {feature_name}")
    print(f"Тип: {feature_series.dtype}")
    print(f"Диапазон: [{feature_series.min():.2f}, {feature_series.max():.2f}]")
    
    return feature_series, feature_name

def perform_binning(feature_series, num_bins=5, labels=None, strategy='equal'):
    """
    Выполнение биннинга числового признака
    Args:
        feature_series (Series): числовой признак для биннинга
        num_bins (int): количество интервалов
        labels (list): метки для категорий
        strategy (str): стратегия биннинга ('equal' - равные интервалы, 
                       'quantile' - равные квантили)
    Returns:
        Series: признак с категориями
    """
    if len(feature_series) < num_bins:
        raise ValueError(f"Количество наблюдений ({len(feature_series)}) меньше количества интервалов ({num_bins})")
    
    if labels is None:
        labels = [f'Bin_{i+1}' for i in range(num_bins)]
    
    if len(labels) != num_bins:
        raise ValueError("Количество меток должно совпадать с количеством интервалов")
    if strategy == 'equal':
        binned_feature = pd.cut(
            feature_series, 
            bins=num_bins, 
            labels=labels,
            include_lowest=True
        )
        print(f"Биннинг выполнен: {num_bins} равных интервалов")
    
    elif strategy == 'quantile':
        binned_feature = pd.qcut(
            feature_series,
            q=num_bins,
            labels=labels,
            duplicates='drop'
        )
        print(f"Биннинг выполнен: {num_bins} квантильных интервалов")
    
    else:
        raise ValueError("Неподдерживаемая стратегия. Используйте: 'equal' или 'quantile'")
    
    return binned_feature

def create_bar_plot(binned_feature, feature_name='feature', figsize=(10, 6)):
    """
    Создание bar plot для визуализации распределения по категориям
    Args:
        binned_feature (Series): признак после биннинга
        feature_name (str): название исходного признака для заголовка
        figsize (tuple): размер графика
    Returns:
        Series: распределение по категориям
    """
    plt.figure(figsize=figsize)
    value_counts = binned_feature.value_counts().sort_index()
    ax = value_counts.plot(kind='bar', color='lightsteelblue', edgecolor='navy', alpha=0.7)
    plt.title(f'Распределение признака "{feature_name}" после биннинга', fontsize=14, pad=20)
    plt.xlabel('Категории', fontsize=12)
    plt.ylabel('Количество наблюдений', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(value_counts):
        ax.text(i, v + max(value_counts) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return value_counts

def evaluate_binning_uniformity(value_counts):
    """
    Оценка равномерности распределения по категориям
    Args:
        value_counts (Series): распределение по категориям
    Returns:
        dict: метрики равномерности
    """
    total_count = value_counts.sum()
    expected_count = total_count / len(value_counts)
    deviations = abs(value_counts - expected_count) / expected_count * 100
    max_deviation = deviations.max()
    mean_deviation = deviations.mean()
    
    metrics = {
        'total_observations': total_count,
        'num_bins': len(value_counts),
        'expected_per_bin': expected_count,
        'max_deviation_percent': max_deviation,
        'mean_deviation_percent': mean_deviation,
        'is_uniform': max_deviation < 20,
        'value_counts': value_counts
    }
    return metrics

def run_complete_binning_pipeline(csv_file_path=None, feature_name=None, num_bins=5, strategy='equal'):
    """
    Полный пайплайн биннинга
    Args:
        csv_file_path (str): путь к CSV файлу (опционально, по умолчанию diabetes)
        feature_name (str): название признака для биннинга
        num_bins (int): количество интервалов
        strategy (str): стратегия биннинга ('equal' или 'quantile')
    Returns:
        tuple: (DataFrame с результатами, метрики, Series биннинг)
    """
    df = load_data(csv_file_path)
    numeric_feature, selected_feature_name = select_numeric_feature(df, feature_name)

    binned_feature = perform_binning(numeric_feature, num_bins, strategy=strategy)
    result_df = df.copy()
    result_df[f'{selected_feature_name}_binned'] = binned_feature
    value_counts = create_bar_plot(binned_feature, selected_feature_name)
    metrics = evaluate_binning_uniformity(value_counts)

    print(f"\nМетрики:")
    print(f"Всего наблюдений: {metrics['total_observations']}")
    print(f"Количество интервалов: {metrics['num_bins']}")
    print(f"Ожидаемое в каждом бине: {metrics['expected_per_bin']:.1f}")
    print(f"Максимальное отклонение: {metrics['max_deviation_percent']:.1f}%")
    print(f"Среднее отклонение: {metrics['mean_deviation_percent']:.1f}%")
    print(f"Равномерное распределение: {'ДА' if metrics['is_uniform'] else 'НЕТ'}")
    
    print(f"\nРАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
    for bin_name, count in value_counts.items():
        percentage = (count / metrics['total_observations']) * 100
        print(f"   {bin_name}: {count} наблюдений ({percentage:.1f}%)")
    
    return result_df, metrics, binned_feature

def save_result_df(result_df, output_path=None):
    """
    Сохранение результата в CSV файл
    Args:
        result_df (DataFrame): DataFrame с результатами биннинга
        output_path (str): путь для сохранения (если None - генерируется автоматически)
    Returns:
        str: путь к сохраненному файлу
    """
    if output_path is None:
        output_path = "binning_result.csv"
    
    result_df.to_csv(output_path, index=False)
    print(f"Результат сохранен в: {output_path}")
    return output_path

if __name__ == "__main__":
    result_df, metrics, binned = run_complete_binning_pipeline(
        # csv_file_path='test_data.csv',
        # feature_name='number',
        num_bins=5,
    )
    save_result_df(result_df, output_path=None)