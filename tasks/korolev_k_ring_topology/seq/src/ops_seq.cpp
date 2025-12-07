#include "korolev_k_ring_topology/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "korolev_k_ring_topology/common/include/common.hpp"

namespace korolev_k_ring_topology {

KorolevKRingTopologySEQ::KorolevKRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KorolevKRingTopologySEQ::ValidationImpl() {
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

bool KorolevKRingTopologySEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KorolevKRingTopologySEQ::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int source = input.source;
  int dest = input.dest;

  int dims[1] = {size};
  int periods[1] = {1};
  int reorder = 0;

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);

  int left_neighbor = 0;
  int right_neighbor = 0;
  MPI_Cart_shift(cart_comm, 0, 1, &left_neighbor, &right_neighbor);

  uint64_t data_size = 0;
  std::vector<int> data;

  const int num_iterations = 50;

  for (int iter = 0; iter < num_iterations; ++iter) {
    if (source == dest) {
      if (rank == source) {
        GetOutput() = input.data;
      }
      data_size = input.data.size();
      MPI_Bcast(&data_size, 1, MPI_UINT64_T, source, cart_comm);
      if (rank != source) {
        GetOutput().resize(data_size);
      }
      MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, source, cart_comm);
      continue;
    }

    int steps_right = (dest - source + size) % size;

    if (rank == source) {
      data = input.data;
      data_size = static_cast<uint64_t>(data.size());

      MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, cart_comm);
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, cart_comm);
    }

    int current_step = (rank - source + size) % size;

    if (current_step > 0 && current_step <= steps_right) {
      MPI_Recv(&data_size, 1, MPI_UINT64_T, left_neighbor, 0, cart_comm, MPI_STATUS_IGNORE);
      data.resize(data_size);
      MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, left_neighbor, 1, cart_comm, MPI_STATUS_IGNORE);

      if (rank == dest) {
        GetOutput() = data;
      } else {
        MPI_Send(&data_size, 1, MPI_UINT64_T, right_neighbor, 0, cart_comm);
        MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, right_neighbor, 1, cart_comm);
      }
    }

    MPI_Bcast(&data_size, 1, MPI_UINT64_T, dest, cart_comm);
    if (rank != dest) {
      GetOutput().resize(data_size);
    }
    MPI_Bcast(GetOutput().data(), static_cast<int>(data_size), MPI_INT, dest, cart_comm);

    for (std::size_t i = 0; i < GetOutput().size(); ++i) {
      GetOutput()[i] += iter;
      GetOutput()[i] -= iter;
    }
  }

  MPI_Comm_free(&cart_comm);

  return true;
}

bool KorolevKRingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace korolev_k_ring_topology
