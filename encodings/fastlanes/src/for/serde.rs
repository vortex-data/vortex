use std::fmt::Formatter;

use vortex_array::{ArrayChildVisitor, ArrayVisitorImpl, DeserializeMetadata, SerializeMetadata};
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::ScalarValue;

use crate::FoRArray;

impl ArrayVisitorImpl<ScalarValueMetadata> for FoRArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded())
    }

    fn _metadata(&self) -> ScalarValueMetadata {
        ScalarValueMetadata(self.reference_scalar().value().clone())
    }
}

#[derive(Clone, Debug)]
pub struct ScalarValueMetadata(ScalarValue);

impl SerializeMetadata for ScalarValueMetadata {
    fn serialize(&self) -> Option<Vec<u8>> {
        Some(self.0.to_flexbytes())
    }
}

impl DeserializeMetadata for ScalarValueMetadata {
    type Output = ScalarValue;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        ScalarValue::from_flexbytes(
            metadata.ok_or_else(|| vortex_err!("Missing ScalarValue metadata"))?,
        )
    }

    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        Self::deserialize(metadata)
            .map(|value| write!(f, "{}", value))
            .unwrap_or_else(|_| write!(f, "<unknown>"))
    }
}
