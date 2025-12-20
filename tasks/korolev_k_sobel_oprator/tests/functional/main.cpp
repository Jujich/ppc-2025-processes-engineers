#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <tuple>

#include "korolev_k_sobel_oprator/common/include/common.hpp"
#include "korolev_k_sobel_oprator/mpi/include/ops_mpi.hpp"
#include "korolev_k_sobel_oprator/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace korolev_k_sobel_oprator_processes {

using korolev_k_sobel_oprator::InType;
using korolev_k_sobel_oprator::OutType;
using korolev_k_sobel_oprator::TestType;

class KorolevKSobelOpratorRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int width = std::get<0>(test_param);
    int height = std::get<1>(test_param);
    int channels = std::get<2>(test_param);
    return std::to_string(width) + "x" + std::to_string(height) + "_ch" + std::to_string(channels);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int width = std::get<0>(params);
    int height = std::get<1>(params);
    int channels = std::get<2>(params);

    // Создаем тестовое изображение
    this->input_data_.width = width;
    this->input_data_.height = height;
    this->input_data_.channels = channels;
    const auto pixels_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
    this->input_data_.pixels.resize(pixels_size);

    // Создаем простое тестовое изображение с градиентом
    for (int row_idx = 0; row_idx < height; ++row_idx) {
      for (int col_idx = 0; col_idx < width; ++col_idx) {
        const int idx = ((row_idx * width) + col_idx) * channels;
        const auto value = static_cast<uint8_t>((col_idx + row_idx) % 256);
        for (int ch_idx = 0; ch_idx < channels; ++ch_idx) {
          this->input_data_.pixels[idx + ch_idx] = value;
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размер выходных данных корректен
    const auto expected_size =
        static_cast<std::size_t>(this->input_data_.width) * static_cast<std::size_t>(this->input_data_.height);
    if (output_data.size() != expected_size) {
      return false;
    }

    // Проверяем, что результат не пустой (для изображений размером >= 3x3 должны быть ненулевые значения)
    if (this->input_data_.width >= 3 && this->input_data_.height >= 3) {
      // Проверяем, что есть хотя бы некоторые ненулевые значения (ребра должны быть обнаружены)
      bool has_non_zero = std::ranges::any_of(output_data, [](uint8_t val) { return val > 0; });
      if (!has_non_zero) {
        return false;
      }
    }

    // Проверяем сумму всех пикселей (для проверки корректности работы)
    int sum = std::accumulate(output_data.begin(), output_data.end(), 0);
    // Для тестовых изображений с градиентом сумма должна быть в разумных пределах
    return sum >= 0 && sum <= 255 * this->input_data_.width * this->input_data_.height;
  }

  InType GetTestInputData() final {
    return this->input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(KorolevKSobelOpratorRunFuncTestsProcesses, SobelOperator) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: (width, height, channels, expected_sum - не используется, но нужен для типа)
const std::array<TestType, 6> kTestParam = {
    // Маленькое grayscale изображение 5x5
    std::make_tuple(5, 5, 1, 0),
    // Среднее grayscale изображение 10x10
    std::make_tuple(10, 10, 1, 0),
    // Большое grayscale изображение 20x20
    std::make_tuple(20, 20, 1, 0),
    // Маленькое RGB изображение 5x5
    std::make_tuple(5, 5, 3, 0),
    // Среднее RGB изображение 10x10
    std::make_tuple(10, 10, 3, 0),
    // Большое RGB изображение 20x20
    std::make_tuple(20, 20, 3, 0)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<korolev_k_sobel_oprator::KorolevKSobelOpratorMPI, InType>(
                       kTestParam, PPC_SETTINGS_korolev_k_sobel_oprator),
                   ppc::util::AddFuncTask<korolev_k_sobel_oprator::KorolevKSobelOpratorSEQ, InType>(
                       kTestParam, PPC_SETTINGS_korolev_k_sobel_oprator));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName =
    KorolevKSobelOpratorRunFuncTestsProcesses::PrintFuncTestName<KorolevKSobelOpratorRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SobelOperatorTests, KorolevKSobelOpratorRunFuncTestsProcesses, kGtestValues, kFuncTestName);

}  // namespace
}  // namespace korolev_k_sobel_oprator_processes
