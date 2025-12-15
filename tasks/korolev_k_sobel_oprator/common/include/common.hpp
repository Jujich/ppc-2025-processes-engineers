#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace korolev_k_sobel_oprator {

// Input: одномерный массив пикселей изображения и его размеры
// Для цветного изображения: channels = 3 (RGB), для grayscale: channels = 1
struct ImageData {
  std::vector<uint8_t> pixels;  // Одномерный массив пикселей
  int width{};                  // Ширина изображения
  int height{};                 // Высота изображения
  int channels{};               // Количество каналов (1 для grayscale, 3 для RGB)
};

using InType = ImageData;
using OutType = std::vector<uint8_t>;  // Результат - одномерный массив с выделенными ребрами

// Для тестов: (width, height, channels, expected_sum)
using TestType = std::tuple<int, int, int, int>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace korolev_k_sobel_oprator
