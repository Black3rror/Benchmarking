#ifdef __cplusplus
    #include <cstdio>
    #include <cstdint>
#else
    #include <stdio.h>
    #include <stdint.h>
#endif
#include "benchmark.h"
#include "benchmarking_utils.h"

#ifdef __cplusplus
    using namespace std;
#endif

/**
 * @brief Function to perform benchmarking.
 *
 * @param repetitions: The number of repetitions to measure the runtime of the model.
 * @param warmups: The number of warmups to perform before measuring the runtime of the model.
 */
void benchmark(int repetitions, int warmups){

    delay_ms(2000);

    print_text("Benchmark start\r\n");

    ModelInput* samples_x;
    ModelOutput* samples_y;
    int n_samples;
    ModelOutput* prediction_ptr;
    uint64_t execution_timer_ticks;
#ifdef TIMER_CLK_FREQ_KHZ
    float execution_time_ms;
    char execution_time_ms_text[50];
#endif
    ModelHandler* model_ptr;

    model_ptr = initialize_model();
    get_samples(&samples_x, &samples_y, &n_samples);

#ifdef CHECK_MODEL_CORRECTNESS
	for (int i=0 ; i<n_samples ; i++){
		prediction_ptr = model_predict(model_ptr, samples_x[i]);
        print_expected_vs_predicted(samples_y[i], *prediction_ptr);
	}
#endif

    for(int i=0 ; i<repetitions ; i++){
        // warmup the model
        for (int j=0 ; j<warmups ; j++){
            prediction_ptr = model_predict(model_ptr, samples_x[0]);
        }

        start_timer();
        prediction_ptr = model_predict(model_ptr, samples_x[0]);
        execution_timer_ticks = stop_timer_and_measure();
#ifndef TIMER_CLK_FREQ_KHZ
        char text[100];
        snprintf(text, sizeof(text), "Execution time: %llu ticks\r\n", execution_timer_ticks);
        print_text(text);
#else
        execution_time_ms = (float)execution_timer_ticks / TIMER_CLK_FREQ_KHZ;
        float_to_string(execution_time_ms_text, sizeof(execution_time_ms_text), execution_time_ms, 5);
        char text[100];

        // since the STM32 GCC compiler has problem with formatting the uint64_t number, we will
        // convert it to text manually. Maximum length of the number can be 18 digits
        uint32_t execution_timer_ticks_h = (uint32_t)(execution_timer_ticks / 1000000000);
        uint32_t execution_timer_ticks_l = (uint32_t)(execution_timer_ticks % 1000000000);
        char execution_timer_ticks_text[20];
        if (execution_timer_ticks_h == 0) {
            snprintf(execution_timer_ticks_text, sizeof(execution_timer_ticks_text), "%u", execution_timer_ticks_l);
        }
        else {
            snprintf(execution_timer_ticks_text, sizeof(execution_timer_ticks_text), "%u%u", execution_timer_ticks_h, execution_timer_ticks_l);
        }

        snprintf(text, sizeof(text), "Execution time: %s ms (%s ticks)\r\n", execution_time_ms_text, execution_timer_ticks_text);
        print_text(text);
#endif

    }

    print_text("Benchmark end\r\n");
}
