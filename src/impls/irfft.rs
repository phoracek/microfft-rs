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

        // The forward operation was:
        // x[0] = (x0.re + x0.im, x0.re - x0.im)
        //
        // Let's name the two components of x[0] as packed.re and packed.im.
        //
        // We can solve sum.re by adding equations of packed.re and packed.im.
        // And we can solve sum.im by substracting the equation of packed.im
        // from packed.re:
        let x0 = x[0];
        x[0] = Complex32::new((x0.re + x0.im) / 2., (x0.re - x0.im) / 2.);

        let u = m / 2;
        for k in 1..u {
            // The forward operation was:
            // 1) sum = (x_k + x_nk)/2
            // 2) diff = (x_k - x_nk)/2
            // 3) x_k.re = sum.re + twiddle_re * sum.im + twiddle_im * diff.re
            // 4) x_k.im = diff.im + twiddle_im * sum.im - twiddle_re * diff.re
            // 5) x_nk.re = sum.re - twiddle_re * sum.im - twiddle_im * diff.re
            // 6) x_nk.im = -diff.im + twiddle_im * sum.im - twiddle_re * diff.re
            //
            // To invert the operation, we will first solve for sum and diff,
            // and then use them to solve for the original x_k and x_nk.

            let s = k * table_stride;
            let twiddle_re = tables::SINE[table_len - s] * -1.; // -cos(2*PI*k/N)
            let twiddle_im = tables::SINE[s - 1]; // sin(2*PI*k/N)

            let (x_k, x_nk) = (x[k], x[m - k]);

            // Let's solve for sum.re and diff.im, this can be done directly by
            // adding equations 3 and 5, and subtracting equations 6 from 4:
            let sum_re = (x_k.re + x_nk.re) / 2.;
            let diff_im = (x_k.im - x_nk.im) / 2.;

            // Now let's solve sum.im and diff.re. These two form a system of
            // linear equations and can be solved using Cramer's rule:
            let a = (x_k.re - x_nk.re) / 2.;
            let b = (x_k.im + x_nk.im) / 2.;
            let twiddle_norm_sq = twiddle_re * twiddle_re + twiddle_im * twiddle_im;
            let sum_im = (a * twiddle_re + b * twiddle_im) / twiddle_norm_sq;
            let diff_re = (a * twiddle_im - b * twiddle_re) / twiddle_norm_sq;

            // Now we can reconstruct x_k and x_nk from sum and diff. This can
            // be done by adding equations 1 and 2, and subtracting 2 from 1:
            let sum = Complex32::new(sum_re, sum_im);
            let diff = Complex32::new(diff_re, diff_im);
            x[k] = sum + diff;
            x[m - k] = sum - diff;
        }

        // The forward operation was:
        // x[u] = (xu.re, -xu.im)
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
