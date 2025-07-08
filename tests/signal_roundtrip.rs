use microfft::test_utils::*;
use microfft::Complex32;
use std::convert::TryInto;

// Macro to generate complex FFT round-trip tests for all signal types
macro_rules! cfft_tests {
    ( $( $name:ident: ($N:expr, $cfft_name:ident, $ifft_name:ident), )* ) => {
        $(
            #[test]
            fn $name() {
                let test_cases = [
                    ("silence", signal_generators::silence_complex($N)),
                    ("noise", signal_generators::noise_complex($N, 12345)),
                    ("sine_full_cycle", signal_generators::sine_complex($N, 1.0, $N as f32)),
                    ("sine", signal_generators::sine_complex($N, 1.7, $N as f32)),
                    ("pulse", signal_generators::single_pulse_complex($N)),
                    ("ramp", signal_generators::ramp_complex($N, 1.7, $N as f32)),
                ];

                for (test_name, input) in test_cases.iter() {
                    let input_array: [Complex32; $N] = input.clone().try_into().unwrap();
                    let original = input_array.clone();

                    // Forward FFT
                    let mut fft_input = input_array.clone();
                    let _fft_result = microfft::complex::$cfft_name(&mut fft_input);

                    // Inverse FFT
                    let recovered = microfft::inverse::$ifft_name(&mut fft_input);

                    // Check that we recover the original signal (within tolerance)
                    for (i, (orig, recovered)) in original.iter().zip(recovered.iter()).enumerate() {
                        assert!(
                            approx::complex_eq(*orig, *recovered, 0.01),
                            "Complex FFT round-trip failed for {} signal at index {}: original={:?}, recovered={:?}, difference={:?}",
                            test_name,
                            i,
                            orig,
                            recovered,
                            Complex32::new(orig.re - recovered.re, orig.im - recovered.im)
                        );
                    }
                }
            }
        )*
    };
}

// Generate complex FFT round-trip tests for all sizes
cfft_tests! {
    cfft_2: (2, cfft_2, ifft_2),
    cfft_4: (4, cfft_4, ifft_4),
    cfft_8: (8, cfft_8, ifft_8),
    cfft_16: (16, cfft_16, ifft_16),
    cfft_32: (32, cfft_32, ifft_32),
    cfft_64: (64, cfft_64, ifft_64),
    cfft_128: (128, cfft_128, ifft_128),
    cfft_256: (256, cfft_256, ifft_256),
    cfft_512: (512, cfft_512, ifft_512),
    cfft_1024: (1024, cfft_1024, ifft_1024),
    cfft_2048: (2048, cfft_2048, ifft_2048),
    cfft_4096: (4096, cfft_4096, ifft_4096),
    cfft_8192: (8192, cfft_8192, ifft_8192),
    cfft_16384: (16384, cfft_16384, ifft_16384),
    cfft_32768: (32768, cfft_32768, ifft_32768),
}

// Macro to generate real FFT round-trip tests for all signal types
macro_rules! rfft_tests {
    ( $( $name:ident: ($N:expr, $rfft_name:ident, $irfft_name:ident), )* ) => {
        $(
            #[test]
            fn $name() {
                let test_cases = [
                    ("silence", signal_generators::silence_real($N)),
                    ("noise", signal_generators::noise_real($N, 12345)),
                    ("sine_full_cycle", signal_generators::sine_real($N, 1.0, $N as f32)),
                    ("sine", signal_generators::sine_real($N, 2.3, $N as f32)),
                    ("pulse", signal_generators::single_pulse_real($N)),
                    ("ramp", signal_generators::ramp_real($N, 1.7, $N as f32)),
                ];

                for (test_name, input) in test_cases.iter() {
                    let input_array: [f32; $N] = input.clone().try_into().unwrap();
                    let original = input_array.clone();

                    // Forward real FFT
                    let mut fft_input = input_array.clone();
                    let mut fft_result = microfft::real::$rfft_name(&mut fft_input);

                    // Inverse real FFT
                    let recovered = microfft::inverse_real::$irfft_name(&mut fft_result);

                    // Check that we recover the original signal (within tolerance)
                    for (i, (orig, recovered)) in original.iter().zip(recovered.iter()).enumerate() {
                        assert!(
                            approx::f32_eq(*orig, *recovered, 0.01),
                            "Real FFT round-trip failed for {} signal at index {}: original={}, recovered={}, difference={}",
                            test_name,
                            i,
                            orig,
                            recovered,
                            orig - recovered
                        );
                    }
                }
            }
        )*
    };
}

// Generate real FFT round-trip tests for all sizes
rfft_tests! {
    rfft_2: (2, rfft_2, irfft_2),
    rfft_4: (4, rfft_4, irfft_4),
    rfft_8: (8, rfft_8, irfft_8),
    rfft_16: (16, rfft_16, irfft_16),
    rfft_32: (32, rfft_32, irfft_32),
    rfft_64: (64, rfft_64, irfft_64),
    rfft_128: (128, rfft_128, irfft_128),
    rfft_256: (256, rfft_256, irfft_256),
    rfft_512: (512, rfft_512, irfft_512),
    rfft_1024: (1024, rfft_1024, irfft_1024),
    rfft_2048: (2048, rfft_2048, irfft_2048),
    rfft_4096: (4096, rfft_4096, irfft_4096),
    rfft_8192: (8192, rfft_8192, irfft_8192),
    rfft_16384: (16384, rfft_16384, irfft_16384),
    rfft_32768: (32768, rfft_32768, irfft_32768),
}
