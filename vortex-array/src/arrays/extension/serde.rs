use crate::arrays::{ExtensionArray, ExtensionEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<ExtensionArray> for ExtensionEncoding {}
