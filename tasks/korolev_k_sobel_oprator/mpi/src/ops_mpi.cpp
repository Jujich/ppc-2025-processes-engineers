#include "korolev_k_sobel_oprator/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
std::vector<uint8_t> ConvertToGrayscale(const std::vector<uint8_t> &pixels, int width, int channels, int start_row,
                                        int num_rows) {
  std::vector<uint8_t> grayscale(static_cast<std::size_t>(width * num_rows));

  if (channels == 1) {
    for (int y = 0; y < num_rows; ++y) {
      int src_y = start_row + y;
      for (int x = 0; x < width; ++x) {
        int src_idx = (src_y * width + x) * channels;
        grayscale[y * width + x] = pixels[src_idx];
      }
    }
    return grayscale;
  }

  for (int y = 0; y < num_rows; ++y) {
    int src_y = start_row + y;
    for (int x = 0; x < width; ++x) {
      int src_idx = (src_y * width + x) * channels;
      uint8_t r = pixels[src_idx];
      uint8_t g = (channels > 1) ? pixels[src_idx + 1] : 0;
      uint8_t b = (channels > 2) ? pixels[src_idx + 2] : 0;
      grayscale[y * width + x] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }
  }
  return grayscale;
}

// Применение оператора Собеля к локальному блоку grayscale изображения
// local_grayscale содержит локальные строки плюс верхнюю и нижнюю граничные строки
std::vector<uint8_t> ApplySobelOperatorLocal(const std::vector<uint8_t> &local_grayscale, int width, int local_height) {
  std::vector<uint8_t> result(static_cast<std::size_t>(width * local_height), 0);

  // Обрабатываем только внутренние пиксели
  // Для оператора Собеля нужны соседние строки, поэтому обрабатываем начиная с y=1 и заканчивая y=local_height-2
  int start_y = 1;
  int end_y = local_height - 1;

  for (int y = start_y; y < end_y; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int gx = 0;
      int gy = 0;

      // Применяем матрицы свертки
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int pixel_idx = (y + ky) * width + (x + kx);
          int pixel_value = static_cast<int>(local_grayscale[pixel_idx]);
          gx += pixel_value * kSobelX[ky + 1][kx + 1];
          gy += pixel_value * kSobelY[ky + 1][kx + 1];
        }
      }

      // Вычисляем величину градиента: |Gx| + |Gy|
      int magnitude = std::abs(gx) + std::abs(gy);

      // Нормализуем в диапазон [0, 255]
      magnitude = std::min(255, magnitude / 4);

      result[y * width + x] = static_cast<uint8_t>(magnitude);
    }
  }

  return result;
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
  std::size_t expected_size = static_cast<std::size_t>(input.width * input.height * input.channels);
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
      GetOutput() = std::vector<uint8_t>(static_cast<std::size_t>(width * height), 0);
    } else {
      GetOutput() = {};
    }
    return true;
  }

  // Распределяем строки между процессами
  const int size_z = static_cast<int>(size);
  const int base_rows = height / size_z;
  const int rem_rows = height % size_z;

  int local_start_row = rank * base_rows + std::min(rank, rem_rows);
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
  std::vector<uint8_t> local_pixels(static_cast<std::size_t>(width * local_rows_with_borders * channels));

  if (rank == 0) {
    // Процесс 0 копирует свои данные
    for (int y = 0; y < local_rows_with_borders; ++y) {
      int src_y = local_start_row_with_border + y;
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          int src_idx = (src_y * width + x) * channels + c;
          int dst_idx = (y * width + x) * channels + c;
          local_pixels[dst_idx] = all_pixels[src_idx];
        }
      }
    }

    // Отправляем данные остальным процессам
    for (int dest = 1; dest < size_z; ++dest) {
      int dest_start_row = dest * base_rows + std::min(dest, rem_rows);
      int dest_num_rows = base_rows + (dest < rem_rows ? 1 : 0);
      int dest_rows_with_borders = dest_num_rows;
      if (dest > 0) {
        dest_rows_with_borders++;
      }
      if (dest < size_z - 1) {
        dest_rows_with_borders++;
      }
      int dest_start_row_with_border = (dest > 0) ? dest_start_row - 1 : dest_start_row;

      int send_count = dest_rows_with_borders * width * channels;
      MPI_Send(all_pixels.data() + dest_start_row_with_border * width * channels, send_count, MPI_UNSIGNED_CHAR, dest,
               0, MPI_COMM_WORLD);
    }
  } else {
    // Принимаем данные от процесса 0
    int recv_count = local_rows_with_borders * width * channels;
    MPI_Recv(local_pixels.data(), recv_count, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Конвертируем локальный блок в grayscale
  std::vector<uint8_t> local_grayscale = ConvertToGrayscale(local_pixels, width, channels, 0, local_rows_with_borders);

  // Применяем оператор Собеля к локальному блоку
  std::vector<uint8_t> local_result = ApplySobelOperatorLocal(local_grayscale, width, local_rows_with_borders);

  // Извлекаем только нужные строки (без граничных)
  std::vector<uint8_t> local_result_clean(static_cast<std::size_t>(width * local_num_rows));
  int offset = (rank > 0) ? 1 : 0;
  for (int y = 0; y < local_num_rows; ++y) {
    for (int x = 0; x < width; ++x) {
      local_result_clean[y * width + x] = local_result[(y + offset) * width + x];
    }
  }

  // Собираем результаты в процесс 0
  if (rank == 0) {
    GetOutput().resize(static_cast<std::size_t>(width * height));
    // Копируем результат процесса 0
    for (int y = 0; y < local_num_rows; ++y) {
      for (int x = 0; x < width; ++x) {
        GetOutput()[y * width + x] = local_result_clean[y * width + x];
      }
    }

    // Принимаем результаты от остальных процессов
    int current_row = local_num_rows;
    for (int src = 1; src < size_z; ++src) {
      int src_num_rows = base_rows + (src < rem_rows ? 1 : 0);
      int recv_count = src_num_rows * width;
      MPI_Recv(GetOutput().data() + current_row * width, recv_count, MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      current_row += src_num_rows;
    }
  } else {
    // Отправляем результат процессу 0
    int send_count = local_num_rows * width;
    MPI_Send(local_result_clean.data(), send_count, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  // Рассылаем результат всем процессам
  if (rank == 0) {
    for (int dest = 1; dest < size_z; ++dest) {
      MPI_Send(GetOutput().data(), width * height, MPI_UNSIGNED_CHAR, dest, 1, MPI_COMM_WORLD);
    }
  } else {
    GetOutput().resize(static_cast<std::size_t>(width * height));
    MPI_Recv(GetOutput().data(), width * height, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

bool KorolevKSobelOpratorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace korolev_k_sobel_oprator
