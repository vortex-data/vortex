use vortex_array::vtable::SerdeVTable;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl SerdeVTable<&DateTimePartsArray> for DateTimePartsEncoding {}
