mod filter;
mod scalar_at;
mod slice;

use crate::ArrayComputeImpl;
use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{FilterKernelAdapter, KernelRef};

impl ArrayComputeImpl for DecimalArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(DecimalEncoding).some();
}
