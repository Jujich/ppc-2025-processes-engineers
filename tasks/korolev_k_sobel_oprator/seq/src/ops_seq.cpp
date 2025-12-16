#include "korolev_k_sobel_oprator/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "korolev_k_sobel_oprator/common/include/common.hpp"

namespace korolev_k_sobel_oprator {

namespace {

// Матрицы Собеля для свертки
constexpr std::array<std::array<int, 3>, 3> kSobelX = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};
constexpr std::array<std::array<int, 3>, 3> kSobelY = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

// Конвертация цветного изображения в grayscale
std::vector<uint8_t> ConvertToGrayscale(const std::vector<uint8_t> &pixels, int width, int height, int channels) {
  if (channels == 1) {
    return pixels;
  }

  const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<uint8_t> grayscale(size);
  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      const int idx = ((row_idx * width) + col_idx) * channels;
      // Формула для конвертации RGB в grayscale: 0.299*R + 0.587*G + 0.114*B
      const uint8_t r = pixels[idx];
      const uint8_t g = (channels > 1) ? pixels[idx + 1] : 0;
      const uint8_t b = (channels > 2) ? pixels[idx + 2] : 0;
      const int gray_idx = (row_idx * width) + col_idx;
      grayscale[gray_idx] = static_cast<uint8_t>((0.299 * r) + (0.587 * g) + (0.114 * b));
    }
  }
  return grayscale;
}

// Применение оператора Собеля к grayscale изображению
std::vector<uint8_t> ApplySobelOperator(const std::vector<uint8_t> &grayscale, int width, int height) {
  const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<uint8_t> result(size, 0);

  // Обрабатываем только внутренние пиксели (пропускаем границы)
  for (int row_idx = 1; row_idx < height - 1; ++row_idx) {
    for (int col_idx = 1; col_idx < width - 1; ++col_idx) {
      int gx = 0;
      int gy = 0;

      // Применяем матрицы свертки
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const int pixel_idx = ((row_idx + ky) * width) + (col_idx + kx);
          const int pixel_value = static_cast<int>(grayscale[pixel_idx]);
          const int kernel_y = ky + 1;
          const int kernel_x = kx + 1;
          gx += pixel_value * kSobelX.at(static_cast<std::size_t>(kernel_y)).at(static_cast<std::size_t>(kernel_x));
          gy += pixel_value * kSobelY.at(static_cast<std::size_t>(kernel_y)).at(static_cast<std::size_t>(kernel_x));
        }
      }

      // Вычисляем величину градиента: |Gx| + |Gy|
      int magnitude = std::abs(gx) + std::abs(gy);

      // Нормализуем в диапазон [0, 255]
      // Максимальное значение для |Gx| + |Gy| при uint8_t: 255 * 4 = 1020
      magnitude = std::min(255, magnitude / 4);

      const int result_idx = (row_idx * width) + col_idx;
      result[result_idx] = static_cast<uint8_t>(magnitude);
    }
  }

  return result;
}

}  // namespace

KorolevKSobelOpratorSEQ::KorolevKSobelOpratorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KorolevKSobelOpratorSEQ::ValidationImpl() {
  const auto &input = GetInput();
  // Проверяем, что размеры корректны и массив пикселей имеет правильный размер
  if (input.width <= 0 || input.height <= 0 || input.channels <= 0) {
    return false;
  }
  const auto expected_size = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height) *
                             static_cast<std::size_t>(input.channels);
  if (input.pixels.size() != expected_size) {
    return false;
  }
  return GetOutput().empty();
}

bool KorolevKSobelOpratorSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KorolevKSobelOpratorSEQ::RunImpl() {
  const auto &input = GetInput();

  // Если изображение слишком маленькое для применения оператора Собеля
  if (input.width < 3 || input.height < 3) {
    const auto size = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
    GetOutput() = std::vector<uint8_t>(size, 0);
    return true;
  }

  // Конвертируем в grayscale, если нужно
  std::vector<uint8_t> grayscale = ConvertToGrayscale(input.pixels, input.width, input.height, input.channels);

  // Применяем оператор Собеля
  GetOutput() = ApplySobelOperator(grayscale, input.width, input.height);

  return true;
}

bool KorolevKSobelOpratorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace korolev_k_sobel_oprator
