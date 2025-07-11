# Benchmarks

This code is used to run benchmarks on an embedded ARM Cortex-M4 system,
specifically the [STM32F3DISCOVERY][1] board.

It measures the number of CPU cycles required to compute complex, real,
inverse complex, and inverse real FFTs of sizes up to 4096.

As a point of comparison, the same benchmarks were originally also performed
against the [Fourier crate][2] which, at that time, was the only other Rust FFT
library with `no_std` support. Unfortunately, the current version of Fourier
refuses to compile without the `std` feature, so it had to be removed from the
benchmark code. The old benchmark results are still included in the table
below.

## Running

To run the benchmarks, make sure the `thumbv7em-none-eabihf` rustc target
and OpenOCD are installed and the board is connected. Then just execute
the `run.py` script.

`run.py` starts OpenOCD in the background. It then builds for every FFT-size
combination a benchmark binary, flashes it onto the board, and runs it.
The results are printed to stdout.

## Results

The following table lists the `microfft` benchmark results from 2025-07-09
together with the `fourier` benchmark results from 2020-03-08.

Measurements are in CPU cycles, so lower is better.

| FFT size | CFFT      | IFFT      | RFFT    | IRFFT   | Fourier (CFFT) |
| -------: | --------: | --------: | ------: | ------: |--------------: |
|    **4** |        69 |        84 |      12 |      11 |            564 |
|    **8** |       199 |       237 |     131 |      49 |          1,462 |
|   **16** |       783 |       880 |     361 |      87 |          2,202 |
|   **32** |     2,295 |     2,823 |   1,151 |     943 |          4,173 |
|   **64** |     6,008 |     7,028 |   3,316 |   4,193 |         10,943 |
|  **128** |    15,400 |    17,548 |   8,269 |  10,005 |         20,904 |
|  **256** |    40,781 |    45,288 |  19,559 |  23,509 |         42,724 |
|  **512** |    89,441 |    99,273 |  50,232 |  58,160 |         97,380 |
| **1024** |   207,596 |   225,658 | 107,959 | 123,136 |          s/o\* |
| **2048** |   468,308 |   500,777 | 243,540 | 274,206 |          s/o\* |
| **4096** | 1,053,366 | 1,128,362 | 538,441 | 607,948 |          s/o\* |

\* FFT cannot be computed due to stack overflow.

[1]: https://www.st.com/en/evaluation-tools/stm32f3discovery.html
[2]: https://crates.io/crates/fourier
