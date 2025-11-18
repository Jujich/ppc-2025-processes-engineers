#include "korolev_k_string_word_count/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "korolev_k_string_word_count/common/include/common.hpp"
#include <mpi.h>

namespace korolev_k_string_word_count {

static int CountWordsChunk(const std::string &s, std::size_t begin, std::size_t end, bool prev_is_space) {
  if (begin >= end) {
    return 0;
  }
  int count = 0;
  bool in_word = false;

  unsigned char first = static_cast<unsigned char>(s[begin]);
  bool first_is_space = std::isspace(first);
  if (!first_is_space && prev_is_space) {
    ++count;
  }
  in_word = !first_is_space;

  for (std::size_t i = begin + 1; i < end; ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    bool is_space = std::isspace(c);
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
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string &s = GetInput();
  const std::size_t n = s.size();

  std::size_t base = n / static_cast<std::size_t>(size);
  std::size_t rem = n % static_cast<std::size_t>(size);

  std::size_t begin =
      static_cast<std::size_t>(rank) * base + std::min<std::size_t>(static_cast<std::size_t>(rank), rem);
  std::size_t end = begin + base + (static_cast<std::size_t>(rank) < rem ? 1 : 0);

  bool prev_is_space = true;
  if (begin > 0) {
    prev_is_space = std::isspace(static_cast<unsigned char>(s[begin - 1]));
  }

  int local_count = CountWordsChunk(s, begin, end, prev_is_space);

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;

  // Had to do some debugging, gonna leave it here in case i need it
  // if (rank == 0) {
  //   std::cout << "mpi:\nstring: " << s << "\nglobal_count: " << global_count << "\n";
  // }

  return true;
}

bool KorolevKStringWordCountMPI::PostProcessingImpl() {
  // Nothing to post-process.
  return true;
}

}  // namespace korolev_k_string_word_count
