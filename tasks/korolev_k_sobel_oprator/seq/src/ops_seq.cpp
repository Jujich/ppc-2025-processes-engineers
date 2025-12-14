#include "korolev_k_sobel_oprator/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "korolev_k_sobel_oprator/common/include/common.hpp"

namespace korolev_k_sobel_oprator {

namespace {

// Матрицы Собеля для свертки
constexpr int kSobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
constexpr int kSobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Конвертация цветного изображения в grayscale
std::vector<uint8_t> ConvertToGrayscale(const std::vector<uint8_t> &pixels, int width, int height, int channels) {
  if (channels == 1) {
    return pixels;
  }

  std::vector<uint8_t> grayscale(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * channels;
      // Формула для конвертации RGB в grayscale: 0.299*R + 0.587*G + 0.114*B
      uint8_t r = pixels[idx];
      uint8_t g = (channels > 1) ? pixels[idx + 1] : 0;
      uint8_t b = (channels > 2) ? pixels[idx + 2] : 0;
      grayscale[y * width + x] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }
  }
  return grayscale;
}

// Применение оператора Собеля к grayscale изображению
std::vector<uint8_t> ApplySobelOperator(const std::vector<uint8_t> &grayscale, int width, int height) {
  std::vector<uint8_t> result(width * height, 0);

  // Обрабатываем только внутренние пиксели (пропускаем границы)
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int gx = 0;
      int gy = 0;

      // Применяем матрицы свертки
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int pixel_idx = (y + ky) * width + (x + kx);
          int pixel_value = static_cast<int>(grayscale[pixel_idx]);
          gx += pixel_value * kSobelX[ky + 1][kx + 1];
          gy += pixel_value * kSobelY[ky + 1][kx + 1];
        }
      }

      // Вычисляем величину градиента: |Gx| + |Gy|
      int magnitude = std::abs(gx) + std::abs(gy);

      // Нормализуем в диапазон [0, 255]
      // Максимальное значение для |Gx| + |Gy| при uint8_t: 255 * 4 = 1020
      magnitude = std::min(255, magnitude / 4);

      result[y * width + x] = static_cast<uint8_t>(magnitude);
    }
  }

  return result;
}

}

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
  std::size_t expected_size = static_cast<std::size_t>(input.width * input.height * input.channels);
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
    GetOutput() = std::vector<uint8_t>(input.width * input.height, 0);
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
