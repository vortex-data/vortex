use crate::arrays::bool::array::BoolArray;
use crate::{encoding_ids, EmptyMetadata, Encoding, EncodingId};

pub struct BoolEncoding;

impl Encoding for BoolEncoding {
    const ID: EncodingId = EncodingId("vortex.bool", encoding_ids::BOOL);
    type Array = BoolArray;
    type Metadata = EmptyMetadata;
}
