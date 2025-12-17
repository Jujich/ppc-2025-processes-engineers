#include "korolev_k_sobel_oprator/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
std::vector<uint8_t> ConvertToGrayscale(const std::vector<uint8_t> &pixels, int width, int channels, int start_row,
                                        int num_rows) {
  const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(num_rows);
  std::vector<uint8_t> grayscale(size);

  if (channels == 1) {
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const int src_y = start_row + row_idx;
      for (int col_idx = 0; col_idx < width; ++col_idx) {
        const int src_idx = ((src_y * width) + col_idx) * channels;
        const int gray_idx = (row_idx * width) + col_idx;
        grayscale[gray_idx] = pixels[src_idx];
      }
    }
    return grayscale;
  }

  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    const int src_y = start_row + row_idx;
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      const int src_idx = ((src_y * width) + col_idx) * channels;
      const uint8_t r = pixels[src_idx];
      const uint8_t g = (channels > 1) ? pixels[src_idx + 1] : 0;
      const uint8_t b = (channels > 2) ? pixels[src_idx + 2] : 0;
      const int gray_idx = (row_idx * width) + col_idx;
      grayscale[gray_idx] = static_cast<uint8_t>((0.299 * r) + (0.587 * g) + (0.114 * b));
    }
  }
  return grayscale;
}

// Применение оператора Собеля к локальному блоку grayscale изображения
// local_grayscale содержит локальные строки плюс верхнюю и нижнюю граничные строки
std::vector<uint8_t> ApplySobelOperatorLocal(const std::vector<uint8_t> &local_grayscale, int width, int local_height) {
  const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(local_height);
  std::vector<uint8_t> result(size, 0);

  // Обрабатываем только внутренние пиксели
  // Для оператора Собеля нужны соседние строки, поэтому обрабатываем начиная с y=1 и заканчивая y=local_height-2
  const int start_y = 1;
  const int end_y = local_height - 1;

  for (int row_idx = start_y; row_idx < end_y; ++row_idx) {
    for (int col_idx = 1; col_idx < width - 1; ++col_idx) {
      int gx = 0;
      int gy = 0;

      // Применяем матрицы свертки
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const int pixel_idx = ((row_idx + ky) * width) + (col_idx + kx);
          const int pixel_value = static_cast<int>(local_grayscale[pixel_idx]);
          const int kernel_y = ky + 1;
          const int kernel_x = kx + 1;
          gx += pixel_value * kSobelX.at(static_cast<std::size_t>(kernel_y)).at(static_cast<std::size_t>(kernel_x));
          gy += pixel_value * kSobelY.at(static_cast<std::size_t>(kernel_y)).at(static_cast<std::size_t>(kernel_x));
        }
      }

      // Вычисляем величину градиента: |Gx| + |Gy|
      int magnitude = std::abs(gx) + std::abs(gy);

      // Нормализуем в диапазон [0, 255]
      magnitude = std::min(255, magnitude / 4);

      const int result_idx = (row_idx * width) + col_idx;
      result[result_idx] = static_cast<uint8_t>(magnitude);
    }
  }

  return result;
}

// Копирование локальных данных процессом 0
void CopyLocalData(int width, int channels, const std::vector<uint8_t> &all_pixels, std::vector<uint8_t> &local_pixels,
                   int local_start_row_with_border, int local_rows_with_borders) {
  for (int row_idx = 0; row_idx < local_rows_with_borders; ++row_idx) {
    const int src_y = local_start_row_with_border + row_idx;
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      for (int ch_idx = 0; ch_idx < channels; ++ch_idx) {
        const int src_idx = (((src_y * width) + col_idx) * channels) + ch_idx;
        const int dst_idx = (((row_idx * width) + col_idx) * channels) + ch_idx;
        local_pixels[dst_idx] = all_pixels[src_idx];
      }
    }
  }
}

// Отправка данных процессу-получателю
void SendDataToProcess(int dest, int size_z, int width, int channels, int base_rows, int rem_rows,
                       const std::vector<uint8_t> &all_pixels) {
  const int dest_start_row = (dest * base_rows) + std::min(dest, rem_rows);
  const int dest_num_rows = base_rows + (dest < rem_rows ? 1 : 0);
  int dest_rows_with_borders = dest_num_rows;
  if (dest > 0) {
    dest_rows_with_borders++;
  }
  if (dest < size_z - 1) {
    dest_rows_with_borders++;
  }
  const int dest_start_row_with_border = (dest > 0) ? dest_start_row - 1 : dest_start_row;

  const int send_count = dest_rows_with_borders * width * channels;
  const auto offset = static_cast<ptrdiff_t>(dest_start_row_with_border) * static_cast<ptrdiff_t>(width) *
                      static_cast<ptrdiff_t>(channels);
  MPI_Send(all_pixels.data() + offset, send_count, MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
}

// Распределение данных по процессам
void DistributeData(int rank, int size_z, int width, int channels, const std::vector<uint8_t> &all_pixels,
                    std::vector<uint8_t> &local_pixels, int local_start_row_with_border, int local_rows_with_borders,
                    int base_rows, int rem_rows) {
  if (rank == 0) {
    // Процесс 0 копирует свои данные
    CopyLocalData(width, channels, all_pixels, local_pixels, local_start_row_with_border, local_rows_with_borders);

    // Отправляем данные остальным процессам
    for (int dest = 1; dest < size_z; ++dest) {
      SendDataToProcess(dest, size_z, width, channels, base_rows, rem_rows, all_pixels);
    }
  } else {
    // Принимаем данные от процесса 0
    const int recv_count = local_rows_with_borders * width * channels;
    MPI_Recv(local_pixels.data(), recv_count, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

// Сбор результатов от всех процессов
void GatherResults(int rank, int size_z, int width, int height, int local_num_rows, int base_rows, int rem_rows,
                   const std::vector<uint8_t> &local_result_clean, std::vector<uint8_t> &output) {
  if (rank == 0) {
    const auto output_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    output.resize(output_size);
    // Копируем результат процесса 0
    for (int row_idx = 0; row_idx < local_num_rows; ++row_idx) {
      for (int col_idx = 0; col_idx < width; ++col_idx) {
        const int output_idx = (row_idx * width) + col_idx;
        const int clean_idx = (row_idx * width) + col_idx;
        output[output_idx] = local_result_clean[clean_idx];
      }
    }

    // Принимаем результаты от остальных процессов
    int current_row = local_num_rows;
    for (int src = 1; src < size_z; ++src) {
      const int src_num_rows = base_rows + (src < rem_rows ? 1 : 0);
      const int recv_count = src_num_rows * width;
      const auto offset = static_cast<ptrdiff_t>(current_row) * static_cast<ptrdiff_t>(width);
      MPI_Recv(output.data() + offset, recv_count, MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_row += src_num_rows;
    }
  } else {
    // Отправляем результат процессу 0
    const int send_count = local_num_rows * width;
    MPI_Send(local_result_clean.data(), send_count, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  }
}

// Рассылка результата всем процессам
void BroadcastResult(int rank, int size_z, int width, int height, std::vector<uint8_t> &output) {
  if (rank == 0) {
    const int total_size = width * height;
    for (int dest = 1; dest < size_z; ++dest) {
      MPI_Send(output.data(), total_size, MPI_UNSIGNED_CHAR, dest, 1, MPI_COMM_WORLD);
    }
  } else {
    const auto output_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    output.resize(output_size);
    const int total_size = width * height;
    MPI_Recv(output.data(), total_size, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

}  // namespace

KorolevKSobelOpratorMPI::KorolevKSobelOpratorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KorolevKSobelOpratorMPI::ValidationImpl() {
  const auto &input = GetInput();
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

bool KorolevKSobelOpratorMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KorolevKSobelOpratorMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> all_pixels;

  // Процесс 0 рассылает размеры изображения
  if (rank == 0) {
    const auto &input = GetInput();
    width = input.width;
    height = input.height;
    channels = input.channels;
    all_pixels = input.pixels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Если изображение слишком маленькое
  if (width < 3 || height < 3) {
    if (rank == 0) {
      const auto output_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
      GetOutput() = std::vector<uint8_t>(output_size, 0);
    } else {
      GetOutput() = {};
    }
    return true;
  }

  // Распределяем строки между процессами
  const int size_z = size;
  const int base_rows = height / size_z;
  const int rem_rows = height % size_z;

  int local_start_row = (rank * base_rows) + std::min(rank, rem_rows);
  int local_num_rows = base_rows + (rank < rem_rows ? 1 : 0);

  // Каждому процессу нужна дополнительная строка сверху и снизу для свертки
  int local_rows_with_borders = local_num_rows;
  if (rank > 0) {
    local_rows_with_borders++;  // Верхняя граница
  }
  if (rank < size_z - 1) {
    local_rows_with_borders++;  // Нижняя граница
  }

  int local_start_row_with_border = (rank > 0) ? local_start_row - 1 : local_start_row;

  // Распределяем данные по процессам
  const auto local_pixels_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(local_rows_with_borders) *
                                 static_cast<std::size_t>(channels);
  std::vector<uint8_t> local_pixels(local_pixels_size);

  DistributeData(rank, size_z, width, channels, all_pixels, local_pixels, local_start_row_with_border,
                 local_rows_with_borders, base_rows, rem_rows);

  // Конвертируем локальный блок в grayscale
  std::vector<uint8_t> local_grayscale = ConvertToGrayscale(local_pixels, width, channels, 0, local_rows_with_borders);

  // Применяем оператор Собеля к локальному блоку
  std::vector<uint8_t> local_result = ApplySobelOperatorLocal(local_grayscale, width, local_rows_with_borders);

  // Извлекаем только нужные строки (без граничных)
  const auto clean_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(local_num_rows);
  std::vector<uint8_t> local_result_clean(clean_size);
  const int offset = (rank > 0) ? 1 : 0;
  for (int row_idx = 0; row_idx < local_num_rows; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      const int clean_idx = (row_idx * width) + col_idx;
      const int result_idx = ((row_idx + offset) * width) + col_idx;
      local_result_clean[clean_idx] = local_result[result_idx];
    }
  }

  // Собираем результаты в процесс 0
  GatherResults(rank, size_z, width, height, local_num_rows, base_rows, rem_rows, local_result_clean, GetOutput());

  // Рассылаем результат всем процессам
  BroadcastResult(rank, size_z, width, height, GetOutput());

  return true;
}

bool KorolevKSobelOpratorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace korolev_k_sobel_oprator
