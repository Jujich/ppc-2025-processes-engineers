#include "korolev_k_string_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

#include "korolev_k_string_word_count/common/include/common.hpp"

namespace korolev_k_string_word_count {

namespace {

int CountWordsChunk(const std::string &s, std::size_t begin, std::size_t end, bool prev_is_space) {
  if (begin >= end) {
    return 0;
  }
  int count = 0;
  bool in_word = false;

  auto first = static_cast<unsigned char>(s[begin]);
  bool first_is_space = std::isspace(first) != 0;
  if (!first_is_space && prev_is_space) {
    ++count;
  }
  in_word = !first_is_space;

  for (std::size_t i = begin + 1; i < end; ++i) {
    auto c = static_cast<unsigned char>(s[i]);
    bool is_space = std::isspace(c) != 0;
    if (!is_space) {
      if (!in_word) {
        ++count;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }
  return count;
}

}  // namespace

KorolevKStringWordCountMPI::KorolevKStringWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KorolevKStringWordCountMPI::ValidationImpl() {
  // Any string is valid.
  return GetOutput() == 0;
}

bool KorolevKStringWordCountMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KorolevKStringWordCountMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string s;
  uint64_t n = 0;

  if (rank == 0) {
    s = GetInput();
    n = static_cast<uint64_t>(s.size());
  }

  MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    s.resize(static_cast<std::size_t>(n));
  }
  if (n > 0) {
    MPI_Bcast(s.data(), static_cast<int>(n), MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  const auto n_size = static_cast<std::size_t>(n);
  const std::size_t base = n_size / static_cast<std::size_t>(size);
  const std::size_t rem = n_size % static_cast<std::size_t>(size);
  const auto rank_z = static_cast<std::size_t>(rank);

  std::size_t begin = (rank_z * base) + std::min(rank_z, rem);
  std::size_t end = begin + base + (rank_z < rem ? std::size_t{1} : std::size_t{0});

  bool prev_is_space = true;
  if (begin > 0) {
    prev_is_space = std::isspace(static_cast<unsigned char>(s[begin - 1])) != 0;
  }

  int local_count = CountWordsChunk(s, begin, end, prev_is_space);

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;

  return true;
}

bool KorolevKStringWordCountMPI::PostProcessingImpl() {
  // Nothing to post-process.
  return true;
}

}  // namespace korolev_k_string_word_count
