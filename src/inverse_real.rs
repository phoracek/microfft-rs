//! Inverse Real FFT (IRFFT)
//!
//! This implementation reverses the optimized real FFT process, taking
//! `N/2` complex frequency-domain samples and producing `N` real
//! time-domain samples. It expects the Nyquist frequency coefficient
//! to be stored in the imaginary part of the DC bin.
//!
//! The inverse real FFT (IRFFT) is optimized by internally using an
//! `N/2`-point inverse CFFT, roughly doubling the computation speed
//! compared to using a full `N`-point inverse CFFT.

use core::convert::TryInto;

use crate::impls::irfft::*;
use crate::Complex32;

macro_rules! irfft_impls {
    ( $( $N:expr => ($irfft_N:ident $(, $feature:expr)?), )* ) => {
        $(
            #[doc = concat!("Perform an in-place ", stringify!($N), "-point inverse RFFT.")]
            #[doc = ""]
            #[doc = "Takes N/2 complex frequency-domain samples and produces N real time-domain samples."]
            #[doc = "The Nyquist frequency coefficient should be stored in the imaginary part of the DC bin."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = ""]
            #[doc = "```"]
            #[doc = concat!("use microfft::{Complex32, inverse_real::", stringify!($irfft_N), "};")]
            #[doc = ""]
            #[doc = concat!("let mut input = [Complex32::default(); ", stringify!($N), " / 2];")]
            #[doc = concat!("let result = ", stringify!($irfft_N), "(&mut input);")]
            #[doc = "```"]
            $( #[cfg(feature = $feature)] )?
            #[inline]
            #[must_use]
            pub fn $irfft_N(input: &mut [Complex32; $N / 2]) -> &mut [f32; $N] {
                IRFftN::<$N>::transform(input).try_into().unwrap()
            }
        )*
    };
}

irfft_impls! {
    2 => (irfft_2),
    4 => (irfft_4),
    8 => (irfft_8, "size-4"),
    16 => (irfft_16, "size-8"),
    32 => (irfft_32, "size-16"),
    64 => (irfft_64, "size-32"),
    128 => (irfft_128, "size-64"),
    256 => (irfft_256, "size-128"),
    512 => (irfft_512, "size-256"),
    1024 => (irfft_1024, "size-512"),
    2048 => (irfft_2048, "size-1024"),
    4096 => (irfft_4096, "size-2048"),
    8192 => (irfft_8192, "size-4096"),
    16384 => (irfft_16384, "size-8192"),
    32768 => (irfft_32768, "size-16384"),
}
