use crate::impls::cfft::*;
use crate::Complex32;

pub(crate) trait IFft {
    type CFft: CFft;

    const N: usize = Self::CFft::N;

    #[inline]
    fn transform(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), Self::N);

        Self::reorder(x);
        Self::CFft::transform(x);
        Self::normalize(x);
    }

    #[inline]
    fn reorder(x: &mut [Complex32]) {
        debug_assert_eq!(x.len(), Self::N);

        let m = Self::N / 2;
        for i in 1..m {
            x.swap(i, Self::N - i);
        }
    }

    #[inline]
    fn normalize(x: &mut [Complex32]) {
        for c in x {
            *c /= Self::N as f32;
        }
    }
}

pub(crate) struct IFftN<const N: usize>;

impl<const N: usize> IFft for IFftN<N>
where
    CFftN<N>: CFft,
{
    type CFft = CFftN<N>;
}
