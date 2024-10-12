This file describes how to benchmark your model on a device.
1. Create a basic project with the timing functionality using any IDE that you prefer.
2. Copy _`benchmark.h`_, _`benchmark.c`_, _`benchmarking_utils.h`_, and _`benchmarking_utils.c`_ to your project.
    - Don't forget to rename `.c` files to `.cpp` or `.cc` if you are using C++.
3. Implement the sections marked with `USER` in _`benchmarking_utils.h`_ and _`benchmarking_utils.c`_.
4. Call the `benchmark` function in your main function.
5. Compile and run your project on the device.
