//! The core Vortex macro to create new encodings and array types.

use std::marker::PhantomData;

use vortex_error::{VortexError, VortexResult};

use crate::encoding::{
    ArrayEncoding, ArrayEncodingExt, ArrayEncodingRef, ArrayMetadataVTable, EncodingId, EncodingRef,
};
use crate::{ArrayData, ArrayMetadata, ArrayTrait, TryDeserializeArrayMetadata};

/// Trait the defines the set of types relating to an array.
/// Because it has associated types it can't be used as a trait object.
pub trait ArrayDef {
    const ID: EncodingId;
    const ENCODING: EncodingRef;

    type Array: ArrayTrait + TryFrom<ArrayData, Error = VortexError>;
    type Metadata: ArrayMetadata + Clone + for<'m> TryDeserializeArrayMetadata<'m>;
    type Encoding: ArrayEncoding + ArrayEncodingExt<D = Self>;
}

/// Typed array wrapper around an array data.
/// TODO(ngates): unwrap TypedArray.
#[derive(Debug, Clone)]
pub struct Array<E> {
    data: ArrayData,
    encoding: PhantomData<E>,
}

impl<E: ArrayEncoding> Array<E> {
    fn metadata(&self) -> Box<dyn ArrayMetadata> {
        self.data.encoding().metadata(&self.data)
    }
}

impl<E> AsRef<ArrayData> for Array<E> {
    fn as_ref(&self) -> &ArrayData {
        &self.data
    }
}

/// Macro to generate all the necessary code for a new type of array encoding. Including:
/// 1. New Array type that implements `AsRef<ArrayData>`, `GetArrayMetadata`, `ToArray`, `IntoArray`, and multiple useful `From`/`TryFrom` implementations.
/// 1. New Encoding type that implements `ArrayEncoding`.
/// 1. New metadata type that implements `ArrayMetadata`.
#[macro_export]
macro_rules! impl_encoding {
    ($id:literal, $code:expr, $Name:ident) => {
        $crate::paste::paste! {
            /// The array definition trait
            #[derive(std::fmt::Debug, Clone)]
            pub struct $Name;
            impl $crate::ArrayDef for $Name {
                const ID: $crate::encoding::EncodingId = $crate::encoding::EncodingId::new($id, $code);
                const ENCODING: $crate::encoding::EncodingRef = &[<$Name Encoding>];
                type Array = [<$Name Array>];
                type Metadata = [<$Name Metadata>];
                type Encoding = [<$Name Encoding>];
            }

            pub type [<$Name Array>] = $crate::Array<$Name>;

            impl [<$Name Array>] {
                #[allow(clippy::same_name_method)]
                fn metadata(&self) -> &[<$Name Metadata>] {
                    self.typed.metadata()
                }

                pub fn len(&self) -> usize {
                    self.typed.array().len()
                }

                pub fn is_empty(&self) -> bool {
                    self.typed.array().is_empty()
                }

                #[allow(dead_code)]
                fn try_from_parts(
                    dtype: vortex_dtype::DType,
                    len: usize,
                    metadata: [<$Name Metadata>],
                    children: std::sync::Arc<[$crate::ArrayData]>,
                    stats: $crate::stats::StatsSet,
                ) -> VortexResult<Self> {
                    Ok(Self { typed: $crate::TypedArray::try_from_parts(dtype, len, metadata, None, children, stats)? })
                }
            }
            impl $crate::GetArrayMetadata for [<$Name Array>] {
                #[allow(clippy::same_name_method)]
                fn metadata(&self) -> std::sync::Arc<dyn $crate::ArrayMetadata> {
                    std::sync::Arc::new(self.metadata().clone())
                }
            }
            impl $crate::ToArrayData for [<$Name Array>] {
                fn to_array(&self) -> $crate::ArrayData {
                    self.typed.to_array()
                }
            }
            impl $crate::IntoArrayData for [<$Name Array>] {
                fn into_array(self) -> $crate::ArrayData {
                    self.typed.into_array()
                }
            }
            impl From<$crate::TypedArray<$Name>> for [<$Name Array>] {
                fn from(typed: $crate::TypedArray<$Name>) -> Self {
                    Self { typed }
                }
            }
            impl TryFrom<$crate::ArrayData> for [<$Name Array>] {
                type Error = vortex_error::VortexError;

                #[inline]
                fn try_from(array: $crate::ArrayData) -> Result<Self, Self::Error> {
                    $crate::TypedArray::<$Name>::try_from(array).map(Self::from)
                }
            }
            impl TryFrom<&$crate::ArrayData> for [<$Name Array>] {
                type Error = vortex_error::VortexError;

                #[inline]
                fn try_from(array: &$crate::ArrayData) -> Result<Self, Self::Error> {
                    $crate::TypedArray::<$Name>::try_from(array).map(Self::from)
                }
            }
            impl From<[<$Name Array>]> for $crate::ArrayData {
                fn from(value: [<$Name Array>]) -> $crate::ArrayData {
                    use $crate::IntoArrayData;
                    value.typed.into_array()
                }
            }

            /// The array encoding
            #[derive(std::fmt::Debug)]
            pub struct [<$Name Encoding>];
            impl $crate::encoding::ArrayEncoding for [<$Name Encoding>] {
                #[inline]
                fn id(&self) -> $crate::encoding::EncodingId {
                    <$Name as $crate::ArrayDef>::ID
                }

                #[inline]
                fn canonicalize(&self, array: $crate::ArrayData) -> vortex_error::VortexResult<$crate::Canonical> {
                    <Self as $crate::encoding::ArrayEncodingExt>::into_canonical(array)
                }

                #[inline]
                fn with_dyn(
                    &self,
                    array: &$crate::ArrayData,
                    f: &mut dyn for<'b> FnMut(&'b (dyn $crate::ArrayTrait + 'b)) -> vortex_error::VortexResult<()>,
                ) -> vortex_error::VortexResult<()> {
                    <Self as $crate::encoding::ArrayEncodingExt>::with_dyn(array, f)
                }
            }
            impl $crate::encoding::ArrayEncodingExt for [<$Name Encoding>] {
                type D = $Name;
            }
            impl $crate::encoding::ArrayMetadataVTable<ArrayData> for [<$Name Encoding>] {
                fn metadata(&self, array: &ArrayData) -> VortexResult<Box<dyn $crate::ArrayMetadata>> {
                    Ok(Box::new([<$Name Metadata>]::try_deserialize_metadata(Some(
                        array.metadata()?.as_ref(),
                    ))?))
                }
            }

            /// Implement ArrayMetadata
            impl $crate::ArrayMetadata for [<$Name Metadata>] {
                #[inline]
                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }

                #[inline]
                fn as_any_arc(self: std::sync::Arc<Self>) -> std::sync::Arc<dyn std::any::Any + std::marker::Send + std::marker::Sync> {
                    self
                }
            }
        }
    };
}

impl<T: AsRef<ArrayData>> ArrayEncodingRef for T {
    fn encoding(&self) -> EncodingRef {
        self.as_ref().encoding()
    }
}

impl AsRef<ArrayData> for ArrayData {
    fn as_ref(&self) -> &ArrayData {
        self
    }
}
