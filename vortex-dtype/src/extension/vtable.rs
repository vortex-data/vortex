// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DType;
use crate::ExtDType;
use crate::ExtID;
use crate::extension::ExtDTypeRef;

// FIXME(ngates): are VTables ZSTs or not?
//  * If yes, then we can create &'static dyn DynVTable references easily.
//  * If no, then we need to manage their lifetimes some other way. And we likely need to hold
//    instances of them on the object itself.
//
// In theory, we could separate out the part that doesn't have an instance (e.g. the ID and the
// deserialize function).

/// The public API for defining new extension DTypes.
pub trait VTable: 'static + Sized + Send + Sync + Clone + Debug {
    /// Associated type containing the deserialized metadata for this extension type
    type Options: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    /// Serialize the options into a byte vector.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Vec<u8>> {
        _ = options;
        vortex_bail!(
            "Serialization not implemented for extension type {}",
            self.id()
        );
    }

    /// Deserialize the options from a byte slice.
    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Options> {
        _ = data;
        vortex_bail!(
            "Deserialization not implemented for extension type {}",
            self.id()
        );
    }

    /// Validate that the given storage type is compatible with this extension type.
    fn validate(&self, options: &Self::Options, storage_dtype: &DType) -> VortexResult<()>;

    // TODO(ngates): add conversion vtable for Arrow extension types.
    // type ArrowConversion: ArrowConversion<Self>;
}

/// A dynamic vtable for extension types, used for type-erased deserialization.
// FIXME(ngates): consider renaming this to ExtDTypePlugin or similar?
pub trait DynVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    /// Deserialize an extension type from serialized options.
    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;

    /// Clones this vtable into a boxed trait object.
    fn clone_box(&self) -> Box<dyn DynVTable>;
}

impl<V: VTable> DynVTable for V {
    fn id(&self) -> ExtID {
        VTable::id(self)
    }

    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let options = VTable::deserialize(self, data)?;
        Ok(ExtDType::try_with_vtable(self.clone(), options, storage_dtype)?.erase())
    }

    fn clone_box(&self) -> Box<dyn DynVTable> {
        Box::new(self.clone())
    }
}

/// An empty options struct for extension dtypes that do not require any options.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmptyOptions;
impl Display for EmptyOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}
