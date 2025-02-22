use vortex_array::vtable::SerdeVTable;

use crate::{ZigZagArray, ZigZagEncoding};

impl SerdeVTable<&ZigZagArray> for ZigZagEncoding {}
