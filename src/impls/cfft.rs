use crate::{tables, Complex32};

pub(crate) trait CFft {
    type Half: CFft;

    const N: usize;
    const LOG2_N: usize = Self::N.ilog2() as usize;

    #[cfg(feature = "bitrev-tables")]
    const BITREV_TABLE: &'static [u16] = tables::BITREV[Self::LOG2_N];

    #[inline]
    fn transform(x: &mut [Complex32]) -> &mut [Complex32] {
        debug_assert_eq!(x.len(), Self::N);

        Self::bit_reverse_reorder(x);
        Self::compute_butterflies(x);
        x
    }

    #[cfg(feature = "bitrev-tables")]
    #[inline]
    fn bit_reverse_reorder(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), Self::N);

        for i in 0..Self::N {
            let j = Self::BITREV_TABLE[i] as usize;
            x.swap(i, j);
        }
    }

    #[cfg(not(feature = "bitrev-tables"))]
    #[inline]
    fn bit_reverse_reorder(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), Self::N);

        let shift = core::mem::size_of::<usize>() * 8 - Self::LOG2_N;
        for i in 0..Self::N {
            let rev = i.reverse_bits();
            let j = rev >> shift;
            if j > i {
                x.swap(i, j);
            }
        }
    }

    #[inline]
    fn compute_butterflies(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), Self::N);

        let m = Self::N / 2;
        let u = m / 2;

        let table_len = tables::SINE.len();
        let table_stride = (table_len + 1) * 4 / Self::N;

        Self::Half::compute_butterflies(&mut x[..m]);
        Self::Half::compute_butterflies(&mut x[m..]);

        // [k = 0] twiddle factor: `1 + 0i`
        let (x_0, x_m) = (x[0], x[m]);
        x[0] = x_0 + x_m;
        x[m] = x_0 - x_m;

        // [k in [1, m/2)] twiddle factor:
        //   - re from SINE table backwards and negative
        //   - im from SINE table directly
        for k in 1..u {
            let s = k * table_stride;
            let re = tables::SINE[table_len - s] * -1.;
            let im = tables::SINE[s - 1];
            let twiddle = Complex32::new(re, im);

            let (x_k, x_km) = (x[k], x[k + m]);
            let y = twiddle * x_km;
            x[k] = x_k + y;
            x[k + m] = x_k - y;
        }

        // [k = m/2] twiddle factor: `0 - 1i`
        let (x_u, x_um) = (x[u], x[u + m]);
        let y = x_um * Complex32::new(0., -1.);
        x[u] = x_u + y;
        x[u + m] = x_u - y;

        // [k in (m/2, m)] twiddle factor:
        //   - re from SINE table directly
        //   - im from SINE table backwards
        for k in (u + 1)..m {
            let s = (k - u) * table_stride;
            let re = tables::SINE[s - 1];
            let im = tables::SINE[table_len - s];
            let twiddle = Complex32::new(re, im);

            let (x_k, x_km) = (x[k], x[k + m]);
            let y = twiddle * x_km;
            x[k] = x_k + y;
            x[k + m] = x_k - y;
        }
    }
}

pub(crate) struct CFftN<const N: usize>;

impl CFft for CFftN<1> {
    type Half = Self;

    const N: usize = 1;

    #[inline]
    fn bit_reverse_reorder(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), 1);
    }

    #[inline]
    fn compute_butterflies(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), 1);
    }
}

impl CFft for CFftN<2> {
    type Half = CFftN<1>;

    const N: usize = 2;

    #[inline]
    fn compute_butterflies(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), 2);

        let (x_0, x_1) = (x[0], x[1]);
        x[0] = x_0 + x_1;
        x[1] = x_0 - x_1;
    }
}

macro_rules! cfft_impls {
    ( $( $N:expr ),* ) => {
        $(
            impl CFft for CFftN<$N> {
                type Half = CFftN<{$N / 2}>;

                const N: usize = $N;
            }
        )*
    };
}

cfft_impls! { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 }
