use vortex_array::vtable::SerdeVTable;

use crate::{RunEndArray, RunEndEncoding};

impl SerdeVTable<&RunEndArray> for RunEndEncoding {}
