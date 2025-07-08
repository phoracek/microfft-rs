use std::convert::TryInto;

use microfft::test_utils::*;

macro_rules! cfft_tests {
    ( $( $name:ident: $N:expr, )* ) => {
        $(
            #[test]
            fn $name() {
                let frequency = 1.7;
                let sample_rate = $N as f32;
                let input = signal_generators::ramp_complex($N, frequency, sample_rate);

                let expected = references::rust_fft(&input);
                let mut input: [_; $N] = input.try_into().unwrap();
                let result = microfft::complex::$name(&mut input);

                approx::assert_complex_eq(result, &expected);
            }
        )*
    };
}

cfft_tests! {
    cfft_2: 2,
    cfft_4: 4,
    cfft_8: 8,
    cfft_16: 16,
    cfft_32: 32,
    cfft_64: 64,
    cfft_128: 128,
    cfft_256: 256,
    cfft_512: 512,
    cfft_1024: 1024,
    cfft_2048: 2048,
    cfft_4096: 4096,
    cfft_8192: 8192,
    cfft_16384: 16384,
    cfft_32768: 32768,
}

macro_rules! rfft_tests {
    ( $( $name:ident: $N:expr, )* ) => {
        $(
            #[test]
            fn $name() {
                let frequency = 1.7;
                let sample_rate = $N as f32;
                let input = signal_generators::ramp_real($N, frequency, sample_rate);

                let expected = references::real_fft(&input);
                let mut input: [_; $N] = input.try_into().unwrap();
                let result = microfft::real::$name(&mut input);

                // The real-valued coefficient at the Nyquist frequency
                // is packed into the imaginary part of the DC bin.
                let coeff_at_nyquist = result[0].im;
                assert_eq!(coeff_at_nyquist, expected[$N / 2].re);
                // Clear this value before checking the results.
                result[0].im = 0.0;
                approx::assert_complex_eq(result, &expected[..($N / 2)]);
            }
        )*
    };
}

rfft_tests! {
    rfft_2: 2,
    rfft_4: 4,
    rfft_8: 8,
    rfft_16: 16,
    rfft_32: 32,
    rfft_64: 64,
    rfft_128: 128,
    rfft_256: 256,
    rfft_512: 512,
    rfft_1024: 1024,
    rfft_2048: 2048,
    rfft_4096: 4096,
    rfft_8192: 8192,
    rfft_16384: 16384,
    rfft_32768: 32768,
}
