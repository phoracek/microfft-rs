//! Test utilities module - only compiled when testing
//!
//! This module contains shared helper functions and utilities used across
//! multiple test files.

/// Approximate floating point comparison utilities
pub mod approx {
    use crate::Complex32;

    /// Check if two complex numbers are approximately equal with given tolerance
    pub fn complex_eq(a: Complex32, b: Complex32, tolerance: f32) -> bool {
        f32_eq(a.re, b.re, tolerance) && f32_eq(a.im, b.im, tolerance)
    }

    /// Check if two f32 values are approximately equal with given tolerance
    pub fn f32_eq(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }

    /// Assert that two slices of complex numbers are approximately equal
    pub fn assert_complex_eq(xa: &[Complex32], xb: &[Complex32]) {
        assert_eq!(xa.len(), xb.len());
        for (a, b) in xa.iter().zip(xb) {
            assert!(complex_eq(*a, *b, 0.005), "{a} !~ {b}");
        }
    }

    /// Assert that two slices of f32 values are approximately equal
    pub fn assert_f32_eq(xa: &[f32], xb: &[f32]) {
        assert_eq!(xa.len(), xb.len());
        for (a, b) in xa.iter().zip(xb) {
            assert!(f32_eq(*a, *b, 0.005), "{a} !~ {b}");
        }
    }
}

/// Reference implementations for comparison testing
pub mod references {
    extern crate std;
    use std::vec::Vec;

    use crate::Complex32;

    /// Reference FFT implementation using rustfft for comparison testing
    pub fn rust_fft(input: &[Complex32]) -> Vec<Complex32> {
        use rustfft::algorithm::Radix4;
        use rustfft::{Fft, FftDirection};

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

    /// Reference real FFT implementation using realfft for comparison testing
    pub fn real_fft(input: &[f32]) -> Vec<Complex32> {
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(input.len());
        let mut input_copy = input.to_vec();
        let mut output = r2c.make_output_vec();

        r2c.process(&mut input_copy, &mut output).unwrap();

        output.iter().map(|c| Complex32::new(c.re, c.im)).collect()
    }
}

/// Signal generators for testing various FFT scenarios
pub mod signal_generators {
    extern crate std;

    use std::f32::consts::PI;
    use std::vec::Vec;

    use crate::Complex32;

    /// Generate a random float in the range [-1.0, 1.0) using a simple LCG
    fn random_float(rng: &mut u32) -> f32 {
        *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((*rng >> 16) as f32 / 32768.0) - 1.0
    }

    /// Generate silence (all zeros) - complex version
    pub fn silence_complex(size: usize) -> Vec<Complex32> {
        std::vec![Complex32::new(0.0, 0.0); size]
    }

    /// Generate silence (all zeros) - real version
    pub fn silence_real(size: usize) -> Vec<f32> {
        silence_complex(size).iter().map(|c| c.re).collect()
    }

    /// Generate white noise using a simple LCG for reproducibility - complex version
    pub fn noise_complex(size: usize, seed: u32) -> Vec<Complex32> {
        let mut rng = seed;
        (0..size)
            .map(|_| {
                let re = random_float(&mut rng);
                let im = random_float(&mut rng);
                Complex32::new(re, im)
            })
            .collect()
    }

    /// Generate white noise using a simple LCG for reproducibility - real version
    pub fn noise_real(size: usize, seed: u32) -> Vec<f32> {
        let mut rng = seed;
        (0..size).map(|_| random_float(&mut rng)).collect()
    }

    /// Generate sine wave with specified frequency - complex version
    pub fn sine_complex(size: usize, frequency: f32, sample_rate: f32) -> Vec<Complex32> {
        (0..size)
            .map(|i| {
                let phase = 2.0 * PI * frequency * (i as f32) / sample_rate;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    /// Generate sine wave with specified frequency - real version
    pub fn sine_real(size: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        sine_complex(size, frequency, sample_rate)
            .iter()
            .map(|c| c.im)
            .collect()
    }

    /// Generate single pulse (impulse) - complex version
    pub fn single_pulse_complex(size: usize) -> Vec<Complex32> {
        let mut signal = std::vec![Complex32::new(0.0, 0.0); size];
        if size > 0 {
            signal[0] = Complex32::new(1.0, 0.0);
        }
        signal
    }

    /// Generate single pulse (impulse) - real version
    pub fn single_pulse_real(size: usize) -> Vec<f32> {
        single_pulse_complex(size).iter().map(|c| c.re).collect()
    }

    /// Generate ramp with specified frequency - complex version
    pub fn ramp_complex(size: usize, frequency: f32, sample_rate: f32) -> Vec<Complex32> {
        (0..size)
            .map(|i| {
                let t = (i as f32) / sample_rate;
                let ramp = (frequency * t) % 1.0;
                Complex32::new(ramp, ramp * 0.5)
            })
            .collect()
    }

    /// Generate ramp with specified frequency - real version
    pub fn ramp_real(size: usize, frequency: f32, sample_rate: f32) -> Vec<f32> {
        ramp_complex(size, frequency, sample_rate)
            .iter()
            .map(|c| c.re)
            .collect()
    }
}
