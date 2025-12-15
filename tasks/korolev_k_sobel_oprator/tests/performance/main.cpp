#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "korolev_k_sobel_oprator/common/include/common.hpp"
#include "korolev_k_sobel_oprator/mpi/include/ops_mpi.hpp"
#include "korolev_k_sobel_oprator/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace korolev_k_sobel_oprator_processes {

using korolev_k_sobel_oprator::InType;
using korolev_k_sobel_oprator::OutType;

class KorolevKSobelOpratorRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kImageSize = 512;  // 512x512 изображение
  static constexpr int kChannels = 3;     // RGB

 protected:
  void SetUp() override {
    // Создаем большое тестовое изображение для performance тестов
    input_data_.width = kImageSize;
    input_data_.height = kImageSize;
    input_data_.channels = kChannels;
    const auto pixels_size = static_cast<std::size_t>(kImageSize) * static_cast<std::size_t>(kImageSize) *
                             static_cast<std::size_t>(kChannels);
    input_data_.pixels.resize(pixels_size);

    // Заполняем изображение градиентом
    for (int row_idx = 0; row_idx < kImageSize; ++row_idx) {
      for (int col_idx = 0; col_idx < kImageSize; ++col_idx) {
        const int idx = ((row_idx * kImageSize) + col_idx) * kChannels;
        const auto value = static_cast<uint8_t>((col_idx + row_idx) % 256);
        for (int ch_idx = 0; ch_idx < kChannels; ++ch_idx) {
          input_data_.pixels[idx + ch_idx] = value;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размер выходных данных корректен
    const auto expected_size = static_cast<std::size_t>(kImageSize) * static_cast<std::size_t>(kImageSize);
    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KorolevKSobelOpratorRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, korolev_k_sobel_oprator::KorolevKSobelOpratorMPI,
                                korolev_k_sobel_oprator::KorolevKSobelOpratorSEQ>(PPC_SETTINGS_korolev_k_sobel_oprator);

const auto kPerfGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KorolevKSobelOpratorRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfSobelOperator, KorolevKSobelOpratorRunPerfTestProcesses, kPerfGtestValues, kPerfTestName);

}  // namespace korolev_k_sobel_oprator_processes
