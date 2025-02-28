#![allow(dead_code)]
#![allow(unused_variables)]
use pyo3::Py;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::stats::StatsSetRef;
use vortex::vtable::VTableRef;
use vortex::{
    ArrayCanonicalImpl, ArrayImpl, ArrayStatisticsImpl, ArrayValidityImpl, ArrayVariantsImpl,
    ArrayVisitorImpl, Canonical, EmptyMetadata,
};

use crate::arrays::py::{PyEncoding, PyEncodingClass};

#[derive(Debug)]
pub struct PyEncodingInstance(Py<PyEncoding>);

impl ArrayImpl for PyEncodingInstance {
    type Encoding = PyEncodingClass;

    fn _len(&self) -> usize {
        todo!()
    }

    fn _dtype(&self) -> &DType {
        todo!()
    }

    fn _vtable(&self) -> VTableRef {
        todo!()
    }
}

impl Clone for PyEncodingInstance {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl ArrayCanonicalImpl for PyEncodingInstance {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }
}

impl ArrayStatisticsImpl for PyEncodingInstance {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        todo!()
    }
}

impl ArrayValidityImpl for PyEncodingInstance {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        todo!()
    }
}

impl ArrayVariantsImpl for PyEncodingInstance {}

impl ArrayVisitorImpl for PyEncodingInstance {
    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
