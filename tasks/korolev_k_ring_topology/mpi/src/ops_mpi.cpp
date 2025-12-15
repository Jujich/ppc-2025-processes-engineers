#include "korolev_k_ring_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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

  if (input.source < 0 || input.source >= size) {
    return false;
  }
  if (input.dest < 0 || input.dest >= size) {
    return false;
  }

  return true;
}

bool KorolevKRingTopologyMPI::PreProcessingImpl() {
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

  int left_neighbor = (rank - 1 + size) % size;
  int right_neighbor = (rank + 1) % size;

  uint64_t data_size = 0;
  std::vector<int> data;

  const int num_iterations = 50;

  for (int iter = 0; iter < num_iterations; ++iter) {
    if (source == dest) {
      if (rank == source) {
        GetOutput() = input.data;
      }
      data_size = input.data.size();
      MPI_Bcast(&data_size, 1, MPI_UINT64_T, source, MPI_COMM_WORLD);
      if (rank != source) {
        GetOutput().resize(data_size);
      }
      MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, source, MPI_COMM_WORLD);
      continue;
    }

    int steps_right = (dest - source + size) % size;

    if (rank == source) {
      data = input.data;
      data_size = static_cast<uint64_t>(data.size());

      MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, MPI_COMM_WORLD);
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, MPI_COMM_WORLD);
    }

    int current_step = (rank - source + size) % size;

    if (current_step > 0 && current_step <= steps_right) {
      MPI_Recv(&data_size, 1, MPI_UINT64_T, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      data.resize(data_size);
      MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, left_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (rank == dest) {
        GetOutput() = data;
      } else {
        MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, MPI_COMM_WORLD);
        MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, MPI_COMM_WORLD);
      }
    }

    MPI_Bcast(&data_size, 1, MPI_UINT64_T, dest, MPI_COMM_WORLD);
    if (rank != dest) {
      GetOutput().resize(data_size);
    }
    MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, dest, MPI_COMM_WORLD);

    for (auto &elem : GetOutput()) {
      elem += iter;
      elem -= iter;
    }
  }

  return true;
}

bool KorolevKRingTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace korolev_k_ring_topology
