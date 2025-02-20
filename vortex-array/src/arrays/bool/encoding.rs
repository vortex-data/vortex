use std::any::Any;
use std::fmt::{Debug, Formatter};

use crate::arrays::bool::array::BoolArray;
use crate::vtable::{ComputeVTable, EncodingVTable};
use crate::{encoding_ids, EmptyMetadata, Encoding, EncodingId};

pub struct BoolEncoding;

impl Encoding for BoolEncoding {
    const ID: EncodingId = EncodingId::new("vortex.bool", encoding_ids::BOOL);

    type Array = BoolArray;
    type Metadata = EmptyMetadata;
}
