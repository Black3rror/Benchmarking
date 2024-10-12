#include "../ei_classifier_porting.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

__attribute__((weak)) EI_IMPULSE_ERROR ei_run_impulse_check_canceled() {
    return EI_IMPULSE_OK;
}

__attribute__((weak)) EI_IMPULSE_ERROR ei_sleep(int32_t time_ms) {
    return EI_IMPULSE_OK;
}

uint64_t ei_read_timer_ms() {
    return 0;
}

uint64_t ei_read_timer_us() {
    return 0;
}

__attribute__((weak)) void ei_printf(const char *format, ...) {
}

__attribute__((weak)) void ei_printf_float(float f) {
}

__attribute__((weak)) void *ei_malloc(size_t size) {
    return malloc(size);
}

__attribute__((weak)) void *ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

__attribute__((weak)) void ei_free(void *ptr) {
    free(ptr);
}

#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C"
#endif
__attribute__((weak)) void DebugLog(const char* s) {
    ei_printf("%s", s);
}
