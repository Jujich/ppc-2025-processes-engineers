#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace korolev_k_ring_topology {

// Входные данные - структура для передачи по кольцу
struct RingMessage {
  int source;              // Процесс-отправитель
  int dest;                // Процесс-получатель
  std::vector<int> data;   // Данные для передачи
};

// Типы для задачи
using InType = RingMessage;
using OutType = std::vector<int>;  // Полученные данные

// Тип для тестов: (source, dest, data, expected_output)
using TestType = std::tuple<int, int, std::vector<int>>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace korolev_k_ring_topology
