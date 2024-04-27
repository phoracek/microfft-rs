//! Inverse FFT (IFFT)

use crate::impls::ifft::*;
use crate::Complex32;

macro_rules! ifft_impls {
    ( $( $N:expr => ($ifft_N:ident $(, $feature:expr)?), )* ) => {
        $(
            #[doc = concat!("Perform an in-place ", stringify!($N), "-point IFFT.")]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = ""]
            #[doc = "```"]
            #[doc = concat!("use microfft::{Complex32, inverse::", stringify!($ifft_N), "};")]
            #[doc = ""]
            #[doc = concat!("let mut input = [Complex32::default(); ", stringify!($N), "];")]
            #[doc = concat!("let result = ", stringify!($ifft_N), "(&mut input);")]
            #[doc = "```"]
            $( #[cfg(feature = $feature)] )?
            #[inline]
            #[must_use]
            pub fn $ifft_N(input: &mut [Complex32; $N]) -> &mut [Complex32; $N] {
                IFftN::<$N>::transform(input);
                input
            }
        )*
    };
}

ifft_impls! {
    2 => (ifft_2),
    4 => (ifft_4, "size-4"),
    8 => (ifft_8, "size-8"),
    16 => (ifft_16, "size-16"),
    32 => (ifft_32, "size-32"),
    64 => (ifft_64, "size-64"),
    128 => (ifft_128, "size-128"),
    256 => (ifft_256, "size-256"),
    512 => (ifft_512, "size-512"),
    1024 => (ifft_1024, "size-1024"),
    2048 => (ifft_2048, "size-2048"),
    4096 => (ifft_4096, "size-4096"),
    8192 => (ifft_8192, "size-8192"),
    16384 => (ifft_16384, "size-16384"),
    32768 => (ifft_32768, "size-32768"),
}
