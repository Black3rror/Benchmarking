#ifndef BENCHMARK_H
#define BENCHMARK_H

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void benchmark(int repetitions, int warmups);

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_H
