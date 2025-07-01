use std::fmt::Display;

use arcref::ArcRef;
use vortex_error::VortexResult;

use crate::ExprRef;

pub type ExprId = ArcRef<str>;
pub type ExprEncodingRef = ArcRef<dyn ExprEncoding>;

/// Encoding trait for a Vortex expression.
///
/// An [`ExprEncoding`] can be registered with a Vortex session in order to support deserialization
/// via the expression protobuf representation.
pub trait ExprEncoding {
    /// Returns the ID of the expression encoding.
    fn id(&self) -> ExprId;

    /// Deserializes an expression from its serialized form.
    ///
    /// Returns `None` if the expression is not serializable.
    fn deserialize(
        &self,
        _options: &[u8],
        _children: Vec<ExprRef>,
    ) -> VortexResult<Option<ExprRef>> {
        Ok(None)
    }
}

impl PartialEq for dyn ExprEncoding {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn ExprEncoding {}

impl Display for dyn ExprEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}
