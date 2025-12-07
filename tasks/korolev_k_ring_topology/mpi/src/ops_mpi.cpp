#include "korolev_k_ring_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "korolev_k_ring_topology/common/include/common.hpp"

namespace korolev_k_ring_topology {

KorolevKRingTopologyMPI::KorolevKRingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KorolevKRingTopologyMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  // Проверяем, что source и dest в допустимых пределах
  if (input.source < 0 || input.source >= size) {
    return false;
  }
  if (input.dest < 0 || input.dest >= size) {
    return false;
  }

  return true;
}

bool KorolevKRingTopologyMPI::PreProcessingImpl() {
  // Инициализируем выходные данные
  GetOutput() = {};
  return true;
}

bool KorolevKRingTopologyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int source = input.source;
  int dest = input.dest;

  // Вычисляем соседей в кольце
  int left_neighbor = (rank - 1 + size) % size;
  int right_neighbor = (rank + 1) % size;

  // Размер данных и сами данные
  uint64_t data_size = 0;
  std::vector<int> data;

  // Если source == dest, просто копируем данные
  if (source == dest) {
    if (rank == source) {
      GetOutput() = input.data;
    }
    // Синхронизируем размер на всех процессах
    data_size = input.data.size();
    MPI_Bcast(&data_size, 1, MPI_UINT64_T, source, MPI_COMM_WORLD);
    if (rank != source) {
      GetOutput().resize(data_size);
    }
    MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, source, MPI_COMM_WORLD);
    return true;
  }

  // Определяем направление передачи (всегда по часовой стрелке - вправо)
  // Вычисляем количество шагов вправо от source до dest
  int steps_right = (dest - source + size) % size;

  // Передаём данные по кольцу
  // Передача идет от source к dest через промежуточные процессы

  if (rank == source) {
    // Отправитель: отправляем данные правому соседу
    data = input.data;
    data_size = static_cast<uint64_t>(data.size());

    MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, MPI_COMM_WORLD);
    MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, MPI_COMM_WORLD);
  }

  // Промежуточные процессы и получатель
  // Определяем, участвует ли текущий процесс в передаче
  int current_step = (rank - source + size) % size;

  if (current_step > 0 && current_step <= steps_right) {
    // Получаем данные от левого соседа
    MPI_Recv(&data_size, 1, MPI_UINT64_T, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    data.resize(data_size);
    MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, left_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == dest) {
      // Получатель: сохраняем данные
      GetOutput() = data;
    } else {
      // Промежуточный процесс: передаём дальше
      MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, MPI_COMM_WORLD);
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, MPI_COMM_WORLD);
    }
  }

  // Распространяем результат на все процессы через Bcast от dest
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, dest, MPI_COMM_WORLD);
  if (rank != dest) {
    GetOutput().resize(data_size);
  }
  MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, dest, MPI_COMM_WORLD);

  return true;
}

bool KorolevKRingTopologyMPI::PostProcessingImpl() {
  // Ничего не требуется
  return true;
}

}  // namespace korolev_k_ring_topology
