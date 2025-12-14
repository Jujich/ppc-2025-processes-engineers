#include <gtest/gtest.h>

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
    input_data_.pixels.resize(static_cast<std::size_t>(kImageSize * kImageSize * kChannels));

    // Заполняем изображение градиентом
    for (int y = 0; y < kImageSize; ++y) {
      for (int x = 0; x < kImageSize; ++x) {
        int idx = (y * kImageSize + x) * kChannels;
        uint8_t value = static_cast<uint8_t>((x + y) % 256);
        for (int c = 0; c < kChannels; ++c) {
          input_data_.pixels[idx + c] = value;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размер выходных данных корректен
    return output_data.size() == static_cast<std::size_t>(kImageSize * kImageSize);
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
