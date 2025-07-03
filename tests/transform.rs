use std::convert::TryInto;

use microfft::Complex32;
use num_complex::ComplexFloat;
use rustfft::algorithm::Radix4;
use rustfft::{Fft, FftDirection};

fn rust_fft(input: &[Complex32]) -> Vec<Complex32> {
    // Convert to rustfft's `num_complex` types, to prevent issues with
    // incompatible versions.
    let mut buf: Vec<_> = input
        .iter()
        .map(|c| rustfft::num_complex::Complex32::new(c.re, c.im))
        .collect();

    let fft = Radix4::new(buf.len(), FftDirection::Forward);
    fft.process(&mut buf);

    buf.iter().map(|c| Complex32::new(c.re, c.im)).collect()
}

fn approx_complex_eq(a: Complex32, b: Complex32) -> bool {
    let abs = a.abs();

    let approx_f32 = |x: f32, y: f32| {
        let diff = (x - y).abs();
        let rel_diff = if abs > 1. { diff / abs } else { diff };
        rel_diff < 0.005
    };

    approx_f32(a.re, b.re) && approx_f32(a.im, b.im)
}

fn assert_approx_complex_eq(xa: &[Complex32], xb: &[Complex32]) {
    assert_eq!(xa.len(), xb.len());
    for (a, b) in xa.iter().zip(xb) {
        assert!(approx_complex_eq(*a, *b), "{a} !~ {b}");
    }
}

fn approx_f32_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

fn assert_approx_f32_eq(xa: &[f32], xb: &[f32]) {
    assert_eq!(xa.len(), xb.len());
    for (a, b) in xa.iter().zip(xb) {
        assert!(approx_f32_eq(*a, *b), "{a} !~ {b}");
    }
}

macro_rules! cfft_tests {
    ( $( $name:ident: $N:expr, )* ) => {
        $(
            #[test]
            fn $name() {
                let input: Vec<_> = (0..$N)
                    .map(|i| i as f32)
                    .map(|f| Complex32::new(f, f))
                    .collect();

                let expected = rust_fft(&input);
                let mut input: [_; $N] = input.try_into().unwrap();
                let result = microfft::complex::$name(&mut input);

                assert_approx_complex_eq(result, &expected);
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

macro_rules! ifft_tests {
    ( $( $name:ident: ($N:expr, $cfft_name:ident), )* ) => {
        $(
            #[test]
            fn $name() {
                let input: Vec<_> = (0..$N)
                    .map(|i| i as f32)
                    .map(|f| Complex32::new(f, f))
                    .collect();
                let mut input: [_; $N] = input.try_into().unwrap();
                let expected = input.clone();

                let transformed = microfft::complex::$cfft_name(&mut input);
                let inversed = microfft::inverse::$name(transformed);

                assert_approx_complex_eq(inversed, &expected);
            }
        )*
    };
}

ifft_tests! {
    ifft_2: (2, cfft_2),
    ifft_4: (4, cfft_4),
    ifft_8: (8, cfft_8),
    ifft_16: (16, cfft_16),
    ifft_32: (32, cfft_32),
    ifft_64: (64, cfft_64),
    ifft_128: (128, cfft_128),
    ifft_256: (256, cfft_256),
    ifft_512: (512, cfft_512),
    ifft_1024: (1024, cfft_1024),
    ifft_2048: (2048, cfft_2048),
    ifft_4096: (4096, cfft_4096),
    ifft_8192: (8192, cfft_8192),
    ifft_16384: (16384, cfft_16384),
    ifft_32768: (32768, cfft_32768),
}

macro_rules! rfft_tests {
    ( $( $name:ident: ($N:expr, $cfft_name:ident), )* ) => {
        $(
            #[test]
            fn $name() {
                let input: Vec<_> = (5..($N+5)).map(|i| i as f32).collect();
                let input_c: Vec<_> = input.iter().map(|f| Complex32::new(*f, 0.)).collect();

                let mut input_c: [_; $N] = input_c.try_into().unwrap();
                let expected = microfft::complex::$cfft_name(&mut input_c);
                let mut input: [_; $N] = input.try_into().unwrap();
                let result = microfft::real::$name(&mut input);
                // The real-valued coefficient at the Nyquist frequency
                // is packed into the imaginary part of the DC bin.
                let coeff_at_nyquist = result[0].im;
                assert_eq!(coeff_at_nyquist, expected[$N / 2].re);
                // Clear this value before checking the results.
                result[0].im = 0.0;
                assert_approx_complex_eq(result, &expected[..($N / 2)]);
            }
        )*
    };
}

rfft_tests! {
    rfft_2: (2, cfft_2),
    rfft_4: (4, cfft_4),
    rfft_8: (8, cfft_8),
    rfft_16: (16, cfft_16),
    rfft_32: (32, cfft_32),
    rfft_64: (64, cfft_64),
    rfft_128: (128, cfft_128),
    rfft_256: (256, cfft_256),
    rfft_512: (512, cfft_512),
    rfft_1024: (1024, cfft_1024),
    rfft_2048: (2048, cfft_2048),
    rfft_4096: (4096, cfft_4096),
    rfft_8192: (8192, cfft_8192),
    rfft_16384: (16384, cfft_16384),
    rfft_32768: (32768, cfft_32768),
}

macro_rules! irfft_tests {
    ( $( $name:ident: ($N:expr, $rfft_name:ident), )* ) => {
        $(
            #[test]
            fn $name() {
                let input: Vec<f32> = (5..($N+5)).map(|i| i as f32).collect();
                let mut input: [f32; $N] = input.clone().try_into().unwrap();
                let expected = input.clone();

                let transformed = microfft::real::$rfft_name(&mut input);
                let inversed = microfft::inverse_real::$name(transformed);

                assert_approx_f32_eq(inversed, &expected);
            }
        )*
    };
}

irfft_tests! {
    irfft_2: (2, rfft_2),
    irfft_4: (4, rfft_4),
    irfft_8: (8, rfft_8),
    irfft_16: (16, rfft_16),
    irfft_32: (32, rfft_32),
    irfft_64: (64, rfft_64),
    irfft_128: (128, rfft_128),
    irfft_256: (256, rfft_256),
    irfft_512: (512, rfft_512),
    irfft_1024: (1024, rfft_1024),
    irfft_2048: (2048, rfft_2048),
    irfft_4096: (4096, rfft_4096),
    irfft_8192: (8192, rfft_8192),
    irfft_16384: (16384, rfft_16384),
    irfft_32768: (32768, rfft_32768),
}
