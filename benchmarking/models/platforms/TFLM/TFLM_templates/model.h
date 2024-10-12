#include <cstdint>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

using namespace std;

#define ARENA_SIZE {arena_size}
#define INPUT_DTYPE {input_dtype}
#define INPUT_N_DIMS {input_n_dims}
{input_dims_size}
#define OUTPUT_DTYPE {output_dtype}
#define OUTPUT_N_DIMS {output_n_dims}
{output_dims_size}
#define N_OPERATORS {n_operators}

extern const unsigned int model_data_size;
extern const unsigned char model_data[];

TfLiteStatus op_resolver(tflite::MicroMutableOpResolver<N_OPERATORS>* resolver_ptr);
