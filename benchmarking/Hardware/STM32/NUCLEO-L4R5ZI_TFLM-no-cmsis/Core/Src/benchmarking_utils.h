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
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model/model.h"


// USER (optional): Comment out this line if you don't want to check the model correctness
#define CHECK_MODEL_CORRECTNESS

// USER (optional): Uncomment the line below and change the value to the frequency of the timer in KHz
#define TIMER_CLK_FREQ_KHZ 120000


// USER: Change the ModelHandler to the type that gets returned by your `initialize_model` function.
typedef tflite::MicroInterpreter ModelHandler;

// USER: Change the ModelInput to the type that gets passed to your `model_predict` function.
// Examples:
//   typedef int8_t ModelInput
//   typedef int8_t ModelInput[28][28]
#if INPUT_N_DIMS == 0
typedef INPUT_DTYPE ModelInput;
#elif INPUT_N_DIMS == 1
typedef INPUT_DTYPE ModelInput[INPUT_DIM_0_SIZE];
#elif INPUT_N_DIMS == 2
typedef INPUT_DTYPE ModelInput[INPUT_DIM_0_SIZE][INPUT_DIM_1_SIZE];
#elif INPUT_N_DIMS == 3
typedef INPUT_DTYPE ModelInput[INPUT_DIM_0_SIZE][INPUT_DIM_1_SIZE][INPUT_DIM_2_SIZE];
#else
#error "Input dimensions greater than 3 are not supported. In case you need more dimensions, please modify the code accordingly"
#endif

// USER: Change the ModelOutput to the type that gets returned by your `model_predict` function.
// Examples:
//   typedef int8_t ModelOutput
//   typedef int8_t ModelOutput[10]
#if OUTPUT_N_DIMS == 0
typedef OUTPUT_DTYPE ModelOutput;
#elif OUTPUT_N_DIMS == 1
typedef OUTPUT_DTYPE ModelOutput[OUTPUT_DIM_0_SIZE];
#elif OUTPUT_N_DIMS == 2
typedef OUTPUT_DTYPE ModelOutput[OUTPUT_DIM_0_SIZE][OUTPUT_DIM_1_SIZE];
#elif OUTPUT_N_DIMS == 3
typedef OUTPUT_DTYPE ModelOutput[OUTPUT_DIM_0_SIZE][OUTPUT_DIM_1_SIZE][OUTPUT_DIM_2_SIZE];
#else
#error "Output dimensions greater than 3 are not supported. In case you need more dimensions, please modify the code accordingly"
#endif


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
