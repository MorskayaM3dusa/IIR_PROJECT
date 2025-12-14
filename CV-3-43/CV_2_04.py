#!/usr/bin/env python3
# -*- coding: utf-8  -*-
# -*- mode:   python -*-

# This file is part of the educational OpenCV tasks.
#
# Copyright (C) 2025  ferrovovan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
CV-2-04: Объединение нескольких масок цветов
Описание задачи:
Объединить маски (на философской основе CV-1-12) для нескольких цветов (например, красный и синий).
"""

import os
import argparse
import cv2
import numpy as np

# Импортируем всё необходимое из CV-1-12
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "../CV-1-12"))
# from pixel_counting import HSV_RED_LOWER_1, HSV_RED_UPPER_1, HSV_RED_LOWER_2, HSV_RED_UPPER_2
# from pixel_counting import load_image, pixel_counting
# from pixel_counting import create_image as create_red_gradient_image


class ColorMask:
	"""
	Контейнер диапазонов HSV для выделения цветов.

	Позволяет хранить несколько диапазонов (например, для красного ―
	0..10 и 170..180 градусов тона), объединять их, создавать итоговую маску.

	Примеры:
		red_mask = ColorMask("red")
		red_mask += (np.array([0, 100, 100]), np.array([10, 255, 255]))
		red_mask += (np.array([170, 100, 100]), np.array([180, 255, 255]))

		mask = red_mask.create_mask(image_hsv)
	"""

	def __init__(self, color_name: str):
		self.name = color_name
		self.hsv_ranges: list[tuple[np.ndarray, np.ndarray]] = []

	def add_hsv_range(self, hsv_range: tuple[np.ndarray, np.ndarray]) -> None:
		"""
		Добавление диапазона HSV.

		Args:
			hsv_range (tuple[np.ndarray, np.ndarray]): нижняя и верхняя границы.

		Raises:
			ValueError: Если диапазон задан неверно.
		"""
		if len(hsv_range) != 2:
			raise ValueError("hsv_range должен содержать (LOWER, UPPER)")
		for value in hsv_range:
			if not isinstance(value, np.ndarray):
				raise TypeError("LOWER и UPPER должны быть np.ndarray")
			if value.shape != (3,):
				raise ValueError("Каждая граница HSV должна быть вектором из 3 чисел")
		self.hsv_ranges.append(hsv_range)

	def __iadd__(self, hsv_range: tuple[np.ndarray, np.ndarray]):
		"""
		Позволяет добавлять диапазоны оператором +=
		"""
		self.add_hsv_range(hsv_range)
		return self

	def create_mask(self, image_hsv: np.ndarray) -> np.ndarray:
		"""
		Создание маски для всех диапазонов цвета.

		Args:
			image_hsv (np.ndarray): Изображение в пространстве HSV.

		Returns:
			np.ndarray: Итоговая бинарная маска.
		"""
		if not self.hsv_ranges:
			raise ValueError(f"У маски {self.name} нет диапазонов HSV")

		masks = [cv2.inRange(image_hsv, lower, upper) for lower, upper in self.hsv_ranges]
		result_mask = masks[0]
		for m in masks[1:]:
			result_mask = cv2.bitwise_or(result_mask, m)
		return result_mask
		


# --- Шаг 1. Создание маски для красного ---
def create_red_mask() -> ColorMask:
	"""
	Создание готовой маски для красного.
	"""
	cm = ColorMask("red")
	cm += (np.array([0, 100, 100]), np.array([10, 255, 255]))
	cm += (np.array([170, 100, 100]), np.array([180, 255, 255]))
	return cm


# --- Шаг 2. Создание маски для синего ---
def create_blue_mask() -> ColorMask:
	"""
	Создание готовой маски для синего.
	"""
	cm = ColorMask("blue")
	cm += (np.array([100, 100, 100]), np.array([130, 255, 255]))
	return cm


# --- Шаг 3. Объединение масок ---
def combine_masks(masks: list[ColorMask], image_hsv: np.ndarray) -> np.ndarray:
	"""
	Создание итоговой маски из нескольких ColorMask.

	Args:
		masks (list[ColorMask]): список масок.
		image_hsv (np.ndarray): HSV-изображение.

	Returns:
		np.ndarray: итоговая бинарная маска.
	"""
	if not masks:
		raise ValueError("Список масок пуст")

	result_mask = masks[0].create_mask(image_hsv)
	for cm in masks[1:]:
		result_mask = cv2.bitwise_or(result_mask, cm.create_mask(image_hsv))
	return result_mask


# --- Шаг 4. Применение маски к изображению ---
def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Применение маски к изображению.

	Args:
		image_bgr (np.ndarray): Исходное изображение в формате BGR.
		mask (np.ndarray): Двоичная маска (одноканальное изображение).

	Returns:
		np.ndarray: Новое изображение, где сохраняются только пиксели,
		соответствующие белым областям маски, остальные зануляются.

	Raises:
		ValueError: Если входное изображение или маска пусты,
		либо если размеры не совпадают.
	"""
	
	if image_bgr is None or image_bgr.size == 0:
		raise ValueError("Входное изображение пустое или None")

	if mask is None or mask.size == 0:
		raise ValueError("Маска пустая или None")

	if image_bgr.shape[:2] != mask.shape[:2]:
		raise ValueError(
			f"Размеры изображения {image_bgr.shape[:2]} "
			f"и маски {mask.shape[:2]} не совпадают"
		)

	# Применение маски: оставляем только белые области, остальные зануляем
	result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
	return result


# --- Шаг 5. Отобразить результат ---
def show_result(original: np.ndarray, mask: np.ndarray, result: np.ndarray) -> None:
	"""
	Отображение исходного изображения, маски и результата.

	Args:
		original (np.ndarray): Исходное изображение в формате BGR.
		mask (np.ndarray): Двоичная маска.
		result (np.ndarray): Изображение после применения маски.

	Returns:
		None: Функция только выводит изображения в окна.

	Raises:
		ValueError: Если одно из изображений пустое.
	"""

	print("Type any key to proceed")
	cv2.imshow("Original", original)
	print("Original Image")
	cv2.waitKey(0)
	cv2.imshow("Combined Mask", mask)
	print("Combined Mask")
	cv2.waitKey(0)
	cv2.imshow("Result", result)
	print("Result Image")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# --- Основная логика ---
def main_process_file(img_path: str) -> None:
	"""
	Основная функция обработки файла-изображения:
	загрузка, создание масок, объединение и отображение.

	Args:
		img_path (str): Путь до входного файла-изображения.

	Returns:
		None: Функция выполняет обработку и отображает результат.

	Raises:
		FileNotFoundError: Если файл по указанному пути отсутствует.
		ValueError: Если изображение не удалось загрузить
			или его формат некорректен.
	"""
	if not os.path.isfile(img_path):
		raise FileNotFoundError(f"Файл '{img_path}' не существует.")

	img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise ValueError(f"Не удалось загрузить изображение: {img_path}")
	img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

	mask_red: ColorMask  = create_red_mask()
	mask_blue: ColorMask = create_blue_mask()
	masks: list[ColorMask] = [mask_red, mask_blue]

	combined_mask: np.ndarray = combine_masks(masks, img_hsv)
	result: np.ndarray   = apply_mask(img_bgr, combined_mask)

	show_result(img_bgr, combined_mask, result)


# --- Пример на HSV-шкале ---
def create_hue_gradient_python(width=360, height=100) -> np.ndarray:
	"""
	Создаёт изображение, где Hue меняется слева направо от 0 до 179.
	S и V = максимум (255).

	Args:
		width (int): ширина изображения (количество оттенков).
		height (int): высота изображения.

	Returns:
		np.ndarray: градиентное изображение в формате BGR.
	"""
	gradient = np.zeros((height, width, 3), dtype=np.uint8)
	for x in range(width):
		hue = int((x / width) * 179)
		gradient[:, x] = (hue, 255, 255)  # HSV
	return cv2.cvtColor(gradient, cv2.COLOR_HSV2BGR)

def create_hue_gradient_numpy(width=360, height=100) -> np.ndarray:
	"""
	Создание градиентного изображения по оттенку (HUE).
	Каждый столбец — значение H от 0 до 179, с фиксированными S=255, V=255.

	Args:
		width (int): ширина изображения (количество оттенков).
		height (int): высота изображения.

	Returns:
		np.ndarray: градиентное изображение в формате BGR.
	"""
	# Генерация линейного градиента HUE от 0 до 179
	hues = np.linspace(0, 179, width, dtype=np.uint8)
	hue_channel = np.tile(hues, (height, 1))

	sat_channel = np.full((height, width), 255, dtype=np.uint8)
	val_channel = np.full((height, width), 255, dtype=np.uint8)

	hsv_image = cv2.merge([hue_channel, sat_channel, val_channel])
	return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def example_synthetic() -> None:
	"""
	Пример на синтетическом изображении:
	Выбираем 
	"""
	img_bgr: np.ndarray = create_hue_gradient_numpy(width=720, height=200)
	img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	
	mask_red: ColorMask  = create_red_mask()
	mask_blue: ColorMask = create_blue_mask()
	masks: list[ColorMask] = [mask_red, mask_blue]

	combined_mask: np.ndarray = combine_masks(masks, img_hsv)
	result: np.ndarray   = apply_mask(img_bgr, combined_mask)

	show_result(img_bgr, combined_mask, result)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="CV-2-04: Объединение масок для красного и синего цветов."
	)

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument(
		"--example_synthetic",
		action="store_true",
		help="Сгенерировать синтетическое изображение с красным, синим, зелёным объектами."
	)
	group.add_argument(
		"--file",
		type=str,
		help="Путь к изображению для обработки."
	)

	args = parser.parse_args()

	if args.example_synthetic:
		example_synthetic()
	elif args.file:
		main_process_file(args.file)

