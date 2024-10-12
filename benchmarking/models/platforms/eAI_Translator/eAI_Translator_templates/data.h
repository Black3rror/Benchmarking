#include <stdint.h>

#define N_SAMPLES {n_samples}
#define SAMPLES_X_SIZE {samples_x_size}
#define SAMPLES_Y_SIZE {samples_y_size}

extern {samples_x_dtype} samples_x[{n_samples}][{samples_x_size}];
extern {samples_y_dtype} samples_y[{n_samples}][{samples_y_size}];
