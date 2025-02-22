use vortex_array::vtable::SerdeVTable;

use crate::{ALPRDArray, ALPRDEncoding};

impl SerdeVTable<&ALPRDArray> for ALPRDEncoding {}
