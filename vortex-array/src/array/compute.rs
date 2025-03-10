use crate::arcref::ArcRef;
use crate::compute::Kernel;

pub trait ComputeKernels {
    const FILTER: Option<ArcRef<dyn Kernel>> = None;
}
