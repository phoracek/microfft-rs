//! FFT on real inputs (RFFT)
//!
//! This implementation takes advantage of the symmetry properties of
//! the FFT to compute an `N`-point RFFT by internally invoking an
//! `N/2`-point CFFT, roughly doubling the computation speed compared
//! to used a full `N`-point CFFT.
//!
//! The produced output is the first half out the output returned by
//! the corresponding `N`-point CFFT, i.e. the real DC value and
//! `N/2 - 1` positive-frequency terms. Additionally, the real-valued
//! coefficient at the Nyquist frequency is packed into the imaginary part
//! of the DC bin. The negative-frequency terms
//! are not computed, since they can be calculated from the
//! positive-frequency terms and are therefore redundant.

use core::convert::TryInto;

use crate::impls::rfft::*;
use crate::Complex32;

macro_rules! rfft_impls {
    ( $( $N:expr => ($rfft_N:ident $(, $feature:expr)?), )* ) => {
        $(
            #[doc = concat!("Perform an in-place ", stringify!($N), "-point RFFT.")]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = ""]
            #[doc = "```"]
            #[doc = concat!("use microfft::real::", stringify!($rfft_N), ";")]
            #[doc = ""]
            #[doc = concat!("let mut input = [0.; ", stringify!($N), "];")]
            #[doc = concat!("let result = ", stringify!($rfft_N), "(&mut input);")]
            #[doc = "```"]
            $( #[cfg(feature = $feature)] )?
            #[inline]
            #[must_use]
            pub fn $rfft_N(input: &mut [f32; $N]) -> &mut [Complex32; $N / 2] {
                RFftN::<$N>::transform(input).try_into().unwrap()
            }
        )*
    };
}

rfft_impls! {
    2 => (rfft_2),
    4 => (rfft_4),
    8 => (rfft_8, "size-4"),
    16 => (rfft_16, "size-8"),
    32 => (rfft_32, "size-16"),
    64 => (rfft_64, "size-32"),
    128 => (rfft_128, "size-64"),
    256 => (rfft_256, "size-128"),
    512 => (rfft_512, "size-256"),
    1024 => (rfft_1024, "size-512"),
    2048 => (rfft_2048, "size-1024"),
    4096 => (rfft_4096, "size-2048"),
    8192 => (rfft_8192, "size-4096"),
    16384 => (rfft_16384, "size-8192"),
    32768 => (rfft_32768, "size-16384"),
}
