// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::marker::PhantomData;

use arcref::ArcRef;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::DType;
use crate::v2::ExtDType;
use crate::v2::ExtDTypeRef;

/// A reference-counted string representing an extension type identifier.
pub type ExtId = ArcRef<str>;

/// The public API for defining new extension DTypes.
pub trait VTable: 'static + Sized + Send + Sync {
    /// Associated type containing the deserialized metadata for this extension type
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// Returns the ID for this extension type.
    fn id(options: &Self::Options) -> ExtId;

    /// Serialize the options into a byte vector.
    fn serialize(options: &Self::Options) -> VortexResult<Vec<u8>>;

    /// Deserialize the options from a byte slice.
    fn deserialize(data: &[u8], session: &VortexSession) -> VortexResult<Self::Options>;

    /// Validate that the given storage type is compatible with this extension type.
    fn validate(options: &Self::Options, storage_dtype: &DType) -> VortexResult<()>;
}

/// A dynamic vtable for extension types, used for type-erased deserialization.
pub trait DynVTable: 'static + Send + Sync + Debug + private::Sealed {
    /// Deserialize an extension type from serialized options.
    fn deserialize(
        &self,
        data: &[u8],
        storage_dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<ExtDTypeRef>;
}

/// Adapter to convert a strongly typed VTable into a DynVTable.
pub struct VTableAdapter<V: VTable>(PhantomData<V>);

impl<V: VTable> Debug for VTableAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", std::any::type_name::<V>())
    }
}

impl<V: VTable> DynVTable for VTableAdapter<V> {
    fn deserialize(
        &self,
        data: &[u8],
        storage_dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<ExtDTypeRef> {
        let options = V::deserialize(data, session)?;
        Ok(ExtDType::<V>::try_new(options, storage_dtype)?.erase())
    }
}

impl<V: VTable> From<V> for &'static dyn DynVTable {
    fn from(_value: V) -> Self {
        const { &VTableAdapter::<V>(PhantomData) }
    }
}

mod private {
    use super::VTableAdapter;
    use crate::VTable;

    pub trait Sealed {}
    impl<V: VTable> Sealed for VTableAdapter<V> {}
}
