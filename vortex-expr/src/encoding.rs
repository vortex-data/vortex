// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use arcref::ArcRef;
use vortex_array::{Array, ArrayRef, DeserializeMetadata};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::metadata::ExprMetadata;
use crate::v2::Expression;
use crate::{ExprRef, IntoExpr, VTable};

pub type ExprId = ArcRef<str>;
pub type ExprEncodingRef = ArcRef<dyn ExprEncoding>;

/// Encoding trait for a Vortex expression.
///
/// An [`ExprEncoding`] can be registered with a Vortex session in order to support deserialization
/// via the expression protobuf representation.
pub trait ExprEncoding: 'static + Send + Sync + Debug + private::Sealed {
    fn as_any(&self) -> &dyn Any;

    /// Returns the ID of the expression encoding.
    fn id(&self) -> ExprId;

    /// Deserializes an expression from its serialized form.
    ///
    /// Returns `None` if the expression is not serializable.
    fn build(&self, metadata: &[u8], children: Vec<ExprRef>) -> VortexResult<ExprRef>;

    /// Validates the metadata and children for this expression encoding.
    fn validate(&self, _metadata: &dyn ExprMetadata, _children: &[Expression]) -> VortexResult<()>;

    /// Computes the return DType of the expression given the metadata and children's DTypes.
    fn return_dtype(
        &self,
        metadata: &dyn ExprMetadata,
        children: &[Expression],
        scope: &DType,
    ) -> VortexResult<DType>;

    /// Evaluates the expression against the given scope array.
    fn evaluate(
        &self,
        metadata: &dyn ExprMetadata,
        children: &[Expression],
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef>;
}

#[repr(transparent)]
pub struct ExprEncodingAdapter<V: VTable>(V::Encoding);

impl<V: VTable> ExprEncoding for ExprEncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExprId {
        V::id(&self.0)
    }

    fn build(&self, metadata: &[u8], children: Vec<ExprRef>) -> VortexResult<ExprRef> {
        let metadata = <V::Metadata as DeserializeMetadata>::deserialize(metadata)?;
        Ok(V::build(&self.0, &metadata, children)?.into_expr())
    }

    fn validate(&self, metadata: &dyn ExprMetadata, children: &[Expression]) -> VortexResult<()> {
        let encoding = self.0.as_opt::<V>().ok_or_else(|| {
            vortex_err!(
                "Mismatched encoding ID {} for VTable {}",
                self.id(),
                std::any::type_name::<V>()
            )
        })?;

        let metadata = metadata
            .as_any()
            .downcast_ref::<V::Metadata>()
            .ok_or_else(|| {
                vortex_err!(
                    "Mismatched metadata type {:?} for VTable {}",
                    metadata.as_any().type_id(),
                    self.id()
                )
            })?;

        V::validate(&encoding, metadata, children)
    }

    fn return_dtype(
        &self,
        metadata: &dyn ExprMetadata,
        children: &[Expression],
        scope: &DType,
    ) -> VortexResult<DType> {
        let encoding = self.0.as_::<V>();
        let metadata = metadata
            .as_any()
            .downcast_ref::<V::Metadata>()
            .vortex_expect("mismatched metadata");
        V::return_dtype2(encoding, metadata, children, scope)
    }

    fn evaluate(
        &self,
        metadata: &dyn ExprMetadata,
        children: &[Expression],
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let result = {
            let encoding = self.0.as_::<V>();
            let metadata = metadata
                .as_any()
                .downcast_ref::<V::Metadata>()
                .vortex_expect("mismatched metadata");
            V::evaluate2(encoding, metadata, children, scope)
        }?;

        #[cfg(debug_assertions)]
        {
            let expected_dtype = self.return_dtype(metadata, children, &scope.dtype())?;
            assert_eq!(
                &expected_dtype,
                result.dtype(),
                "Evaluated dtype does not match expected dtype"
            );
            assert_eq!(
                result.len(),
                scope.len(),
                "Evaluated array length does not match scope length"
            );
        }

        Ok(result)
    }
}

impl<V: VTable> Debug for ExprEncodingAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprEncoding")
            .field("id", &self.id())
            .finish()
    }
}

impl Display for dyn ExprEncoding + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn ExprEncoding + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn ExprEncoding + '_ {}

impl dyn ExprEncoding + '_ {
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    pub fn as_<V: VTable>(&self) -> &V::Encoding {
        self.as_opt::<V>()
            .vortex_expect("ExprEncoding is not of the expected type")
    }

    pub fn as_opt<V: VTable>(&self) -> Option<&V::Encoding> {
        self.as_any()
            .downcast_ref::<ExprEncodingAdapter<V>>()
            .map(|e| &e.0)
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for ExprEncodingAdapter<V> {}
}
