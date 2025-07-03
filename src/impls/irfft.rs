use core::slice;

use static_assertions::{assert_eq_align, assert_eq_size};

use crate::impls::ifft::*;
use crate::{tables, Complex32};

pub(crate) trait IRFft {
    type IFft: IFft;

    const N: usize = Self::IFft::N * 2;

    #[inline]
    fn transform(x: &mut [Complex32]) -> &mut [f32] {
        debug_assert_eq!(x.len(), Self::N / 2);

        Self::recombine(x);
        Self::IFft::transform(x);
        Self::unpack_real(x)
    }

    #[inline]
    fn recombine(x: &mut [Complex32]) {
        let m = Self::N / 2;
        debug_assert_eq!(x.len(), m);

        let table_len = tables::SINE.len();
        let table_stride = (table_len + 1) * 4 / Self::N;

        let x0 = x[0];
        x[0] = Complex32::new((x0.re + x0.im) / 2., (x0.re - x0.im) / 2.);

        let u = m / 2;
        for k in 1..u {
            let s = k * table_stride;
            let twiddle_re = tables::SINE[table_len - s] * -1.; // cos(2*PI*k/N)
            let twiddle_im = tables::SINE[s - 1]; // -sin(2*PI*k/N)

            let (x_k, x_nk) = (x[k], x[m - k]);
            let sum = (x_k + x_nk) / 2.;
            let diff = (x_k - x_nk) / 2.;

            x[k] = Complex32::new(
                sum.re - twiddle_re * sum.im - twiddle_im * diff.re,
                diff.im + twiddle_im * sum.im - twiddle_re * diff.re,
            );
            x[m - k] = Complex32::new(
                sum.re + twiddle_re * sum.im + twiddle_im * diff.re,
                -diff.im + twiddle_im * sum.im - twiddle_re * diff.re,
            );
        }

        let xu = x[u];
        x[u] = Complex32::new(xu.re, -xu.im);
    }

    #[inline]
    fn unpack_real(x: &mut [Complex32]) -> &mut [f32] {
        assert_eq_size!(Complex32, [f32; 2]);
        assert_eq_align!(Complex32, f32);
        debug_assert_eq!(x.len(), Self::N / 2);

        let data = x.as_mut_ptr().cast::<f32>();
        unsafe { slice::from_raw_parts_mut(data, Self::N) }
    }
}

pub(crate) struct IRFftN<const N: usize>;

impl IRFft for IRFftN<2> {
    type IFft = IFftN<1>;

    #[inline]
    fn recombine(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), 1);

        let x0 = x[0];
        x[0] = Complex32::new((x0.re + x0.im) / 2., (x0.re - x0.im) / 2.);
    }
}

macro_rules! irfft_impls {
    ( $( $N:expr ),* ) => {
        $(
            impl IRFft for IRFftN<$N> {
                type IFft = IFftN<{$N / 2}>;
            }
        )*
    };
}

irfft_impls! { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 }
