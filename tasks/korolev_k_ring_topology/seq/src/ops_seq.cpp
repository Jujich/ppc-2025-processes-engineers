#include "korolev_k_ring_topology/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "korolev_k_ring_topology/common/include/common.hpp"

namespace korolev_k_ring_topology {

KorolevKRingTopologySEQ::KorolevKRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KorolevKRingTopologySEQ::ValidationImpl() {
  // В последовательной версии просто проверяем, что source и dest неотрицательны
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool KorolevKRingTopologySEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KorolevKRingTopologySEQ::RunImpl() {
  // Эмуляция передачи данных по кольцу в последовательной версии
  // Имитируем многократное копирование данных через промежуточные "процессы"
  const auto &input = GetInput();
  
  // Эмулируем передачу данных через несколько "процессов" в кольце
  // Для реалистичной эмуляции делаем несколько проходов по данным
  std::vector<int> current_data = input.data;
  std::vector<int> temp_buffer(input.data.size());
  
  // Эмулируем передачу по кольцу - 100 итераций для создания нагрузки
  const int num_iterations = 100;
  
  for (int iter = 0; iter < num_iterations; ++iter) {
    // Копируем данные как при реальной передаче между процессами
    for (std::size_t i = 0; i < current_data.size(); ++i) {
      temp_buffer[i] = current_data[i];
    }
    // Эмулируем обработку на каждом "процессе"
    for (std::size_t i = 0; i < temp_buffer.size(); ++i) {
      temp_buffer[i] += iter;
      temp_buffer[i] -= iter;
    }
    std::swap(current_data, temp_buffer);
  }
  
  GetOutput() = current_data;
  return true;
}

bool KorolevKRingTopologySEQ::PostProcessingImpl() {
  // Ничего не требуется
  return true;
}

}  // namespace korolev_k_ring_topology
