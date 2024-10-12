#ifdef __cplusplus
    #include <cstdio>
    #include <cstdint>
    #include <cmath>
#else
    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>
#endif
#include "benchmarking_utils.h"

// USER: You can include libraries here
#include "main.h"
#include "model/model.h"
#include "model/data.h"


#ifdef __cplusplus
    using namespace std;
#endif


#define PRETRAINED 1
#define INCREMENTAL 2


extern TIM_HandleTypeDef htim2;


// USER: Implement the get_samples function here
/**
 * Gets the samples to be used for benchmarking. At least 1 sample must be returned.
 * The first sample is used for runtime measurements. All samples are used for model
 * correctness checking.
 *
 * @param samples_input: The input samples. It will be filled by the function.
 * @param samples_output: The output samples. It will be filled by the function.
 * @param n_samples: The number of samples. It will be filled by the function.
 */
void get_samples(ModelInput** samples_input, ModelOutput** samples_output, int* n_samples){
    *samples_input = samples_x;
    *samples_output = samples_y;

    *n_samples = N_SAMPLES;
}


// USER: Implement the initialize_model function here
/**
 * Initializes the model.
 *
 * @return ModelHandler: The handler to the model.
 */
ModelHandler* initialize_model(void){
    static bool is_initialized = false;
    if (is_initialized) {
        print_text("Model already initialized\r\n");
        exit(1);
    }

    ek_status_t status = EK_SUCCESS;
    static char model_arena[ARENA_SIZE];

    static ModelHandler model;
    #if MODEL_TYPE == PRETRAINED
    model = ek_model_load(model_data, model_arena, ARENA_SIZE, &status);
    #elif MODEL_TYPE == INCREMENTAL
    model = ek_inc_model_load(model_data, model_arena, ARENA_SIZE, &status);
    #endif
    EK_ASSERT(status == EK_SUCCESS);

    is_initialized = true;

    return &model;
}


// USER: Implement the model_predict function here
/**
 * Performs a prediction on the model.
 *
 * @param model: The handler to the model.
 * @param input: The input to the model.
 * @return ModelOutput: The prediction of the model.
 */
ModelOutput* model_predict(ModelHandler* model, ModelInput input){
    static ModelOutput* prediction;

    #if MODEL_TYPE == PRETRAINED
    prediction = ek_predict(*model, input);
    #elif MODEL_TYPE == INCREMENTAL
    prediction = ek_inc_predict(*model, input);
    #endif

    return prediction;
}


// USER: Implement the delay_ms function here
/**
 * Delays the execution for the specified number of milliseconds.
 *
 * @param ms: The number of milliseconds to delay.
 *
 * @note: The exact delay is not important and it can be approximate. So, following is
 * a dirty implementation of delay. Better if you replace it with a more accurate one.
 */
void delay_ms(uint32_t ms){
    HAL_Delay(ms);
}


// USER: Implement the start_timer function here
/**
 * Starts the timer.
 */
void start_timer(void){
    HAL_TIM_Base_Start(&htim2);
}


// USER: Implement the stop_timer_and_measure function here
/**
 * Stops the timer and returns the number of ticks elapsed.
 *
 * @return uint64_t: The number of ticks elapsed.
 */
uint64_t stop_timer_and_measure(void){
    HAL_TIM_Base_Stop(&htim2);
    uint32_t elapsed_ticks = __HAL_TIM_GET_COUNTER(&htim2);
    __HAL_TIM_SET_COUNTER(&htim2, 0);
    return elapsed_ticks;
}


/**
 * Converts a float to a string with a specified precision.
 *
 * @param str: The buffer to store the string.
 * @param size: The size of the buffer.
 * @param value: The float value to convert.
 * @param precision: The number of decimal places to include.
*/
void float_to_string(char *str, size_t size, float value, int precision){
    char intPartStr[20], fracPartStr[20];

    int intPart = (int)value;

    // Extract and round the fractional part
    float fractionalPart = fabs(value - intPart);
    float multiplier = pow(10, precision);
    int fracPart = (int)(round(fractionalPart * multiplier));

    // Check for rounding that affects the integer part
    if (fracPart >= multiplier) {
        fracPart -= (int)multiplier;
        intPart += (value >= 0) ? 1 : -1;
    }

    snprintf(intPartStr, 20, "%d", intPart);
    snprintf(fracPartStr, 20, "%0*d", precision, fracPart);

    if (precision > 0) {
        snprintf(str, size, "%s.%s", intPartStr, fracPartStr);
    } else {
        snprintf(str, size, "%s", intPartStr);
    }
}


// USER: Implement the print_text function here
/**
 * Prints the text to the output.
 *
 * @param text: The text to print.
 */
void print_text(char* text){
    printf(text);
}


#ifdef CHECK_MODEL_CORRECTNESS
// USER: Implement the print_expected_vs_predicted function here
/**
 * The print should end with a new line. For each number in the output, the print should be
 * in the format "[{y_expected}, {y_predicted}]". For example, if the expected output
 * is [1, 2, 3] and the predicted output is [4, 5, 6], the print can be like:
 * "[y_expected, y_predicted]: [1, 4], [2, 5], [3, 6]\n".
 *
 * @param expected: The expected output.
 * @param predicted: The predicted output.
 */
void print_expected_vs_predicted(ModelOutput expected, ModelOutput predicted){
    char expected_str[50], predicted_str[50], text[150];
    float_to_string(expected_str, sizeof(expected_str), (float)expected[0], 5);
    float_to_string(predicted_str, sizeof(predicted_str), (float)predicted[0], 5);
    snprintf(text, sizeof(text), "[y_expected, y_predicted]: [%s, %s]\r\n", expected_str, predicted_str);
    print_text(text);
}
#endif
