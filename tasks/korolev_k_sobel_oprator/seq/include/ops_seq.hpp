#pragma once

#include "korolev_k_sobel_oprator/common/include/common.hpp"
#include "task/include/task.hpp"

namespace korolev_k_sobel_oprator {

class KorolevKSobelOpratorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KorolevKSobelOpratorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace korolev_k_sobel_oprator
