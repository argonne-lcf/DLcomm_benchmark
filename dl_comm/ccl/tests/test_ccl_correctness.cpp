#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>
#if __has_include("../ccl_correctness.hpp")
#include "../ccl_correctness.hpp"
#define REQUIRE(x) do { if (!(x)) { std::cerr << "FAIL line " << __LINE__ << ": " #x "\n"; return 1; } } while (0)
int main(int argc, char** argv) {
  using namespace dlcomm_correctness;
#ifdef DLCOMM_WHOLE_OUTPUT_VERIFIER
  if (argc == 2 && std::string(argv[1]) == "--inject-tail") {
    Expected expected("allreduce", 4097, 2, 1, 6);
    std::vector<float> output(expected.output_count());
    for (size_t j = 0; j < output.size(); ++j) output[j] = expected.value(j);
    output.back() += 1.0f;
    const auto result = verify(output.data(), output.size(), expected);
    std::cout << "CPU INJECTION checks=" << result.checks << " failures=" << result.failures
              << " first_index=" << result.first_index << "\n";
    return result.failures ? 1 : 0;
  }
#else
  (void)argc; (void)argv;
#endif
  REQUIRE(input_value(0, 0, 0) == 1.0f);
  REQUIRE(input_value(1, 0, 0) == 18.0f);
  REQUIRE(input_value(0, 1, 0) == 32.0f);
  REQUIRE(input_value(0, 0, 1) == 128.0f);
  for (int r = 0; r < 16; ++r)
    for (size_t j = 0; j < 4096; ++j)
      for (int i = 0; i < 3; ++i) {
        float v = input_value(r, j, i);
        REQUIRE(v >= 1 && v <= 1021 && v == static_cast<int>(v));
      }
  std::cout << "PASS exact rank/index/iteration patterns\n";
#ifndef DLCOMM_WHOLE_OUTPUT_VERIFIER
  std::cerr << "FAIL missing whole-output verifier\n";
  return 1;
#else
  const char* ops[] = {"allreduce", "allgather", "broadcast", "reduce", "reduce_scatter", "sendrecv", "alltoall"};
  for (const char* name : ops) {
    for (int world : {1, 2, 3, 7}) for (int rank = 0; rank < world; ++rank) {
      const size_t count = 23 * world;
      for (int iteration : {0, 1, 1020, 1021}) {
        Expected expected(name, count, world, rank, iteration);
        std::vector<float> out(expected.output_count());
        for (size_t j = 0; j < out.size(); ++j) {
          float oracle = 0;
          std::string op(name);
          if (op == "allreduce" || op == "reduce" || op == "reduce_scatter") {
            size_t input_index = j + (op == "reduce_scatter" ? rank * (count / world) : 0);
            for (int src = 0; src < world; ++src) oracle += input_value(src, input_index, iteration);
          } else if (op == "allgather") oracle = input_value(j / count, j % count, iteration);
          else if (op == "broadcast") oracle = input_value(0, j, iteration);
          else if (op == "sendrecv") oracle = input_value(1 - rank, j, iteration);
          else oracle = input_value(j / (count / world), rank * (count / world) + j % (count / world), iteration);
          out[j] = oracle;
          REQUIRE(expected.value(j) == oracle);
        }
        auto clean = verify(out.data(), out.size(), expected);
        REQUIRE(clean.checks == out.size() && clean.failures == 0);
        if (!out.empty()) {
          out.back() += 1;
          auto corrupt = verify(out.data(), out.size(), expected);
          REQUIRE(corrupt.checks == out.size() && corrupt.failures == 1);
          REQUIRE(corrupt.first_index == out.size() - 1);
          out.back() = std::numeric_limits<float>::quiet_NaN();
          REQUIRE(verify(out.data(), out.size(), expected).failures == 1);
          out.back() = std::numeric_limits<float>::infinity();
          REQUIRE(verify(out.data(), out.size(), expected).failures == 1);
        }
      }
    }
    std::cout << "PASS whole-output " << name << " including tail/NaN/Inf corruption\n";
  }
  bool rejected = false;
  try { Expected invalid("alltoall", 10, 3, 0, 0); } catch (const std::invalid_argument&) { rejected = true; }
  REQUIRE(rejected);
  rejected = false;
  try { Expected invalid("unknown", 10, 2, 0, 0); } catch (const std::invalid_argument&) { rejected = true; }
  REQUIRE(rejected);
  Expected large("allreduce", 1, max_exact_world, 0, 0);
  std::uint64_t sum = 0;
  for (int src = 0; src < max_exact_world; ++src) sum += static_cast<unsigned>(input_value(src, 0, 0));
  REQUIRE(large.value(0) == static_cast<float>(sum));
  std::cout << "PASS partition rejection and exact-world boundary\n";
#endif
}
#else
int main() { std::cerr << "FAIL missing CPU correctness implementation\n"; return 1; }
#endif
