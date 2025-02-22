use std::any::Any;
use std::fmt::{Debug, Formatter};

use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::bool::array::BoolArray;
use crate::serde::ArrayParts;
use crate::vtable::{ComputeVTable, EncodingVTable, SerdeVTable};
use crate::{encoding_ids, ArrayRef, EmptyMetadata, Encoding, EncodingId};

pub struct BoolEncoding;

impl Encoding for BoolEncoding {
    const ID: EncodingId = EncodingId::new("vortex.bool", encoding_ids::BOOL);

    type Array = BoolArray;
    type Metadata = EmptyMetadata;
}
