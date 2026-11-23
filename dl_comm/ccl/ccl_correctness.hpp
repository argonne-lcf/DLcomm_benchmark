#ifndef DLCOMM_CCL_CORRECTNESS_HPP
#define DLCOMM_CCL_CORRECTNESS_HPP
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
namespace dlcomm_correctness {
// Positive integer floats. Each summand <=1021; world <=16432 guarantees
// every possible partial sum <=2^24, so reduction order cannot change bits.
// Modulo arithmetic avoids overflow for large indices/iteration numbers.
constexpr int max_exact_world = 16432;
inline float input_value(int rank, std::size_t index, int iteration) {
  return static_cast<float>(1 + ((static_cast<std::uint64_t>(rank) % 1021) * 17 +
                                 (index % 1021) * 31 +
                                 (static_cast<std::uint64_t>(iteration) % 1021) * 127) % 1021);
}
// The output contract is independent of vendor headers and GPU execution.
class Expected {
 public:
  Expected(const std::string& op, std::size_t count, int world, int rank, int iteration)
      : op_(op), count_(count), world_(world), rank_(rank), iteration_(iteration) {
    if (world < 1 || world > max_exact_world || rank < 0 || rank >= world ||
        iteration < 0 || count == 0)
      throw std::invalid_argument("invalid correctness shape or exact-world bound");
    if (op != "allreduce" && op != "allgather" && op != "broadcast" &&
        op != "reduce" && op != "reduce_scatter" && op != "sendrecv" && op != "alltoall")
      throw std::invalid_argument("unsupported correctness operation");
    if ((op == "alltoall" || op == "reduce_scatter") && count % world != 0)
      throw std::invalid_argument("partitioned operation requires count divisible by world");
    if (op == "allgather" && count > std::numeric_limits<std::size_t>::max() / world)
      throw std::invalid_argument("allgather output overflow");
    if (op == "allreduce" || op == "reduce" || op == "reduce_scatter") {
      for (std::size_t j = 0; j < sums_.size(); ++j) {
        unsigned sum = 0;
        for (int src = 0; src < world; ++src)
          sum += static_cast<unsigned>(input_value(src, j, iteration));
        sums_[j] = static_cast<float>(sum);
      }
    }
  }
  std::size_t output_count() const {
    if ((op_ == "reduce" && rank_ != 0) ||
        (op_ == "sendrecv" && (world_ < 2 || rank_ > 1))) return 0;
    if (op_ == "allgather") return count_ * world_;
    if (op_ == "reduce_scatter") return count_ / world_;
    return count_;
  }
  float value(std::size_t index) const {
    if (index >= output_count()) throw std::out_of_range("expected output index");
    if (op_ == "allreduce" || op_ == "reduce") return sums_[index % 1021];
    if (op_ == "reduce_scatter") return sums_[(rank_ * (count_ / world_) + index) % 1021];
    if (op_ == "allgather") return input_value(index / count_, index % count_, iteration_);
    if (op_ == "broadcast") return input_value(0, index, iteration_);
    if (op_ == "sendrecv") return input_value(1 - rank_, index, iteration_);
    const std::size_t per = count_ / world_;
    return input_value(index / per, rank_ * per + index % per, iteration_);
  }
 private:
  std::string op_;
  std::size_t count_;
  int world_, rank_, iteration_;
  std::array<float, 1021> sums_{};
};
struct Verification {
  unsigned long long checks = 0, failures = 0;
  std::size_t first_index = 0;
  float first_got = 0, first_expected = 0;
};
inline Verification verify(const float* output, std::size_t count, const Expected& expected) {
  if (count != expected.output_count() || (count && !output))
    throw std::invalid_argument("verification output shape mismatch");
  Verification result;
  for (std::size_t j = 0; j < count; ++j) {
    const float want = expected.value(j);
    ++result.checks;
    // Exact equality also rejects NaN and infinity (expected is finite).
    if (output[j] != want) {
      if (result.failures == 0) {
        result.first_index = j;
        result.first_got = output[j];
        result.first_expected = want;
      }
      ++result.failures;
    }
  }
  return result;
}
} // namespace dlcomm_correctness
#define DLCOMM_WHOLE_OUTPUT_VERIFIER 1
#endif
