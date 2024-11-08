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
#include "tensorflow/lite/micro/system_setup.h"
#include "model/model.h"
#include "model/data.h"


#ifdef __cplusplus
    using namespace std;
#endif


// USER
// We want INPUT_DTYPE_NAME to be one of the names defined in TfLitePtrUnion

// We will define different types as numbers to be able to do the `#if` comparison
// and undefine them afterwards
#define int32_t 1
#define uint32_t 2
#define int64_t 3
#define uint64_t 4
#define float 5
#define TfLiteFloat16 6
#define double 7
#define char 8
// #define const char 9
#define uint8_t 10
#define bool 11
#define int16_t 12
#define uint16_t 13
#define TfLiteComplex64 14
#define TfLiteComplex128 15
#define int8_t 16

#if INPUT_DTYPE == int32_t
#define INPUT_DTYPE_NAME i32
#elif INPUT_DTYPE == uint32_t
#define INPUT_DTYPE_NAME u32
#elif INPUT_DTYPE == int64_t
#define INPUT_DTYPE_NAME i64
#elif INPUT_DTYPE == uint64_t
#define INPUT_DTYPE_NAME u64
#elif INPUT_DTYPE == float
#define INPUT_DTYPE_NAME f
#elif INPUT_DTYPE == TfLiteFloat16
#define INPUT_DTYPE_NAME f16
#elif INPUT_DTYPE == double
#define INPUT_DTYPE_NAME f64
#elif INPUT_DTYPE == char
#define INPUT_DTYPE_NAME raw
// #elif INPUT_DTYPE == const char
// #define INPUT_DTYPE_NAME raw_const
#elif INPUT_DTYPE == uint8_t
#define INPUT_DTYPE_NAME uint8
#elif INPUT_DTYPE == bool
#define INPUT_DTYPE_NAME b
#elif INPUT_DTYPE == int16_t
#define INPUT_DTYPE_NAME i16
#elif INPUT_DTYPE == uint16_t
#define INPUT_DTYPE_NAME ui16
#elif INPUT_DTYPE == TfLiteComplex64
#define INPUT_DTYPE_NAME c64
#elif INPUT_DTYPE == TfLiteComplex128
#define INPUT_DTYPE_NAME c128
#elif INPUT_DTYPE == int8_t
#define INPUT_DTYPE_NAME int8
#else
#error "INPUT_DTYPE not supported"
#endif

// The names defined in TfLitePtrUnion
#if OUTPUT_DTYPE == int32_t
#define OUTPUT_DTYPE_NAME i32
#elif OUTPUT_DTYPE == uint32_t
#define OUTPUT_DTYPE_NAME u32
#elif OUTPUT_DTYPE == int64_t
#define OUTPUT_DTYPE_NAME i64
#elif OUTPUT_DTYPE == uint64_t
#define OUTPUT_DTYPE_NAME u64
#elif OUTPUT_DTYPE == float
#define OUTPUT_DTYPE_NAME f
#elif OUTPUT_DTYPE == TfLiteFloat16
#define OUTPUT_DTYPE_NAME f16
#elif OUTPUT_DTYPE == double
#define OUTPUT_DTYPE_NAME f64
#elif OUTPUT_DTYPE == char
#define OUTPUT_DTYPE_NAME raw
// #elif OUTPUT_DTYPE == const char
// #define OUTPUT_DTYPE_NAME raw_const
#elif OUTPUT_DTYPE == uint8_t
#define OUTPUT_DTYPE_NAME uint8
#elif OUTPUT_DTYPE == bool
#define OUTPUT_DTYPE_NAME b
#elif OUTPUT_DTYPE == int16_t
#define OUTPUT_DTYPE_NAME i16
#elif OUTPUT_DTYPE == uint16_t
#define OUTPUT_DTYPE_NAME ui16
#elif OUTPUT_DTYPE == TfLiteComplex64
#define OUTPUT_DTYPE_NAME c64
#elif OUTPUT_DTYPE == TfLiteComplex128
#define OUTPUT_DTYPE_NAME c128
#elif OUTPUT_DTYPE == int8_t
#define OUTPUT_DTYPE_NAME int8
#else
#error "OUTPUT_DTYPE not supported"
#endif

// We don't need these anymore
#undef int32_t
#undef uint32_t
#undef int64_t
#undef uint64_t
#undef float
#undef TfLiteFloat16
#undef double
#undef char
// #undef const char
#undef uint8_t
#undef bool
#undef int16_t
#undef uint16_t
#undef TfLiteComplex64
#undef TfLiteComplex128
#undef int8_t


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

    tflite::InitializeTarget();

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    static const tflite::Model* model_raw = tflite::GetModel(model_data);
    if (model_raw->version() != TFLITE_SCHEMA_VERSION) {
        char text[100];
        snprintf(text, sizeof(text), "Model provided is schema version %d not equal to supported version %d.\r\n", model_raw->version(), TFLITE_SCHEMA_VERSION);
        print_text(text);
        exit(1);
    }

    // Pull in necessary operation implementations
    TfLiteStatus status;
    static tflite::MicroMutableOpResolver<N_OPERATORS> resolver;
    status = op_resolver(&resolver);
    if (status != kTfLiteOk) {
        print_text("Could not initialize op resolver\r\n");
        exit(1);
    }

    // Build an interpreter to run the model with. We call it model to be compatible with the rest of the code.
    static uint8_t tensor_arena[ARENA_SIZE];
    static ModelHandler model(model_raw, resolver, tensor_arena, ARENA_SIZE);

    // Allocate memory from the tensor_arena for the model's tensors.
    status = model.AllocateTensors();
    if (status != kTfLiteOk) {
        print_text("AllocateTensors() failed\r\n");
        exit(1);
    }

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
    // Place the input value in the model's input tensor
    #if INPUT_N_DIMS == 0
    model->input(0)->data.INPUT_DTYPE_NAME[0] = input;
    #elif INPUT_N_DIMS == 1
    for (int i = 0; i < INPUT_DIM_0_SIZE; i++) {
        model->input(0)->data.INPUT_DTYPE_NAME[i] = input[i];
    }
    #elif INPUT_N_DIMS == 2
    for (int i = 0; i < INPUT_DIM_0_SIZE; i++) {
        for (int j = 0; j < INPUT_DIM_1_SIZE; j++) {
            model->input(0)->data.INPUT_DTYPE_NAME[i*INPUT_DIM_1_SIZE+j] = input[i][j];
        }
    }
    #elif INPUT_N_DIMS == 3
    for (int i = 0; i < INPUT_DIM_0_SIZE; i++) {
        for (int j = 0; j < INPUT_DIM_1_SIZE; j++) {
            for (int k = 0; k < INPUT_DIM_2_SIZE; k++) {
                model->input(0)->data.INPUT_DTYPE_NAME[i*INPUT_DIM_1_SIZE*INPUT_DIM_2_SIZE+j*INPUT_DIM_2_SIZE+k] = input[i][j][k];
            }
        }
    }
    #endif

    // Run inference, and report any error
    TfLiteStatus invoke_status = model->Invoke();
    if (invoke_status != kTfLiteOk) {
        print_text("Invoke failed\r\n");
        exit(1);
    }

    // Obtain the output from model's output tensor
    static ModelOutput output;
    #if OUTPUT_N_DIMS == 0
    output = model->output(0)->data.OUTPUT_DTYPE_NAME[0];
    #elif OUTPUT_N_DIMS == 1
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        output[i] = model->output(0)->data.OUTPUT_DTYPE_NAME[i];
    }
    #elif OUTPUT_N_DIMS == 2
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        for (int j = 0; j < OUTPUT_DIM_1_SIZE; j++) {
            output[i][j] = model->output(0)->data.OUTPUT_DTYPE_NAME[i*OUTPUT_DIM_1_SIZE+j];
        }
    }
    #elif OUTPUT_N_DIMS == 3
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        for (int j = 0; j < OUTPUT_DIM_1_SIZE; j++) {
            for (int k = 0; k < OUTPUT_DIM_2_SIZE; k++) {
                output[i][j][k] = model->output(0)->data.OUTPUT_DTYPE_NAME[i*OUTPUT_DIM_1_SIZE*OUTPUT_DIM_2_SIZE+j*OUTPUT_DIM_2_SIZE+k];
            }
        }
    }
    #endif

    return &output;
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
    char expected_str[50], predicted_str[50], text[150];

    #if OUTPUT_N_DIMS == 0
    float_to_string(expected_str, sizeof(expected_str), (float)expected, 5);
    float_to_string(predicted_str, sizeof(predicted_str), (float)predicted, 5);
    snprintf(text, sizeof(text), "[y_expected, y_predicted]: [%s, %s]\r\n", expected_str, predicted_str);
    print_text(text);

    #elif OUTPUT_N_DIMS == 1
    print_text("[y_expected, y_predicted]: [");
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        float_to_string(expected_str, sizeof(expected_str), (float)expected[i], 5);
        float_to_string(predicted_str, sizeof(predicted_str), (float)predicted[i], 5);
        snprintf(text, sizeof(text), "[%s, %s]", expected_str, predicted_str);
        print_text(text);
        if (i < OUTPUT_DIM_0_SIZE - 1) {
            print_text(", ");
        }
    }
    print_text("]\r\n");

    #elif OUTPUT_N_DIMS == 2
    print_text("[y_expected, y_predicted]: [");
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        print_text("[");
        for (int j = 0; j < OUTPUT_DIM_1_SIZE; j++) {
            float_to_string(expected_str, sizeof(expected_str), (float)expected[i][j], 5);
            float_to_string(predicted_str, sizeof(predicted_str), (float)predicted[i][j], 5);
            snprintf(text, sizeof(text), "[%s, %s]", expected_str, predicted_str);
            print_text(text);
            if (j < OUTPUT_DIM_1_SIZE - 1) {
                print_text(", ");
            }
        }
        print_text("]");
        if (i < OUTPUT_DIM_0_SIZE - 1) {
            print_text(", ");
        }
    }
    print_text("]\r\n");

    #elif OUTPUT_N_DIMS == 3
    print_text("[y_expected, y_predicted]: [");
    for (int i = 0; i < OUTPUT_DIM_0_SIZE; i++) {
        print_text("[");
        for (int j = 0; j < OUTPUT_DIM_1_SIZE; j++) {
            print_text("[");
            for (int k = 0; k < OUTPUT_DIM_2_SIZE; k++) {
                float_to_string(expected_str, sizeof(expected_str), (float)expected[i][j][k], 5);
                float_to_string(predicted_str, sizeof(predicted_str), (float)predicted[i][j][k], 5);
                snprintf(text, sizeof(text), "[%s, %s]", expected_str, predicted_str);
                print_text(text);
                if (k < OUTPUT_DIM_2_SIZE - 1) {
                    print_text(", ");
                }
            }
            print_text("]");
            if (j < OUTPUT_DIM_1_SIZE - 1) {
                print_text(", ");
            }
        }
        print_text("]");
        if (i < OUTPUT_DIM_0_SIZE - 1) {
            print_text(", ");
        }
    }
    print_text("]\r\n");
    #endif
}
#endif
