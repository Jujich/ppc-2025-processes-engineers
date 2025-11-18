#include "korolev_k_string_word_count/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "korolev_k_string_word_count/common/include/common.hpp"

namespace korolev_k_string_word_count {

KorolevKStringWordCountSEQ::KorolevKStringWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KorolevKStringWordCountSEQ::ValidationImpl() {
  // Any string is valid. Ensure output is reset.
  return GetOutput() == 0;
}

bool KorolevKStringWordCountSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KorolevKStringWordCountSEQ::RunImpl() {
  const std::string &s = GetInput();
  if (s.empty()) {
    GetOutput() = 0;
    return true;
  }

  int count = 0;
  bool in_word = false;
  for (unsigned char c : s) {
    if (!std::isspace(c)) {
      if (!in_word) {
        ++count;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }

  GetOutput() = count;
  return true;
}

bool KorolevKStringWordCountSEQ::PostProcessingImpl() {
  // Nothing to post-process.
  return true;
}

}  // namespace korolev_k_string_word_count
