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


#ifdef __cplusplus
    using namespace std;
#endif


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

}


// USER: Implement the initialize_model function here
/**
 * Initializes the model.
 *
 * @return ModelHandler: The handler to the model.
 */
ModelHandler* initialize_model(void){

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
    volatile uint32_t i;
    for (i = 0; i < ms * 10000; i++) {}
}


// USER: Implement the start_timer function here
/**
 * Starts the timer.
 */
void start_timer(void){

}


// USER: Implement the stop_timer_and_measure function here
/**
 * Stops the timer and returns the number of ticks elapsed.
 *
 * @return uint64_t: The number of ticks elapsed.
 */
uint64_t stop_timer_and_measure(void){

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

}
#endif
