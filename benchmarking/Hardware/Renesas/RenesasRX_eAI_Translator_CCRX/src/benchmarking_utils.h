#ifndef BENCHMARKING_UTIL_H
#define BENCHMARKING_UTIL_H

#ifdef __cplusplus
    #include <cstdint>
    #include <cstddef>

    using namespace std;
#else
    #include <stdint.h>
    #include <stddef.h>
#endif

// USER
#include "data.h"
#include "Typedef.h"


// USER (optional): Comment out this line if you don't want to check the model correctness
#define CHECK_MODEL_CORRECTNESS

// USER (optional): Uncomment the line below and change the value to the frequency of the timer in KHz
#define TIMER_CLK_FREQ_KHZ 120000


// USER: Change the ModelHandler to the type that gets returned by your `initialize_model` function.
typedef void* ModelHandler;

// USER: Change the ModelInput to the type that gets passed to your `model_predict` function.
// Examples:
//   typedef int8_t ModelInput
//   typedef int8_t ModelInput[28][28]
typedef TsIN ModelInput[SAMPLES_X_SIZE];

// USER: Change the ModelOutput to the type that gets returned by your `model_predict` function.
// Examples:
//   typedef int8_t ModelOutput
//   typedef int8_t ModelOutput[10]
typedef TsOUT ModelOutput[SAMPLES_Y_SIZE];


void get_samples(ModelInput** samples_input, ModelOutput** samples_output, int* n_samples);
ModelHandler* initialize_model(void);
ModelOutput* model_predict(ModelHandler* model, ModelInput input);

void delay_ms(uint32_t ms);
void start_timer(void);
uint64_t stop_timer_and_measure(void);

void float_to_string(char *str, size_t size, float value, int precision);
void print_text(char* text);
#ifdef CHECK_MODEL_CORRECTNESS
void print_expected_vs_predicted(ModelOutput expected, ModelOutput predicted);
#endif


#endif // BENCHMARKING_UTIL_H
