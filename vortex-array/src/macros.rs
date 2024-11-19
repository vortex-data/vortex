//! The core Vortex macro to create new encodings and array types.

use std::marker::PhantomData;

use vortex_error::{VortexError, VortexResult};

use crate::encoding::{
    ArrayEncoding, ArrayEncodingExt, ArrayEncodingRef, ArrayMetadataVTable, EncodingId, EncodingRef,
};
use crate::{
    ArrayData, ArrayMetadata, ArrayTrait, IntoArrayData, ToArrayData, TryDeserializeArrayMetadata,
};

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
    fn metadata(&self) -> VortexResult<Box<dyn ArrayMetadata>> {
        self.data.encoding().metadata(&self.data)
    }
}

impl<E> ToArrayData for Array<E> {
    fn to_array(&self) -> ArrayData {
        self.data.clone()
    }
}

impl<E> IntoArrayData for Array<E> {
    fn into_array(self) -> ArrayData {
        self.data
    }
}

impl<E> From<Array<E>> for ArrayData {
    fn from(array: Array<E>) -> ArrayData {
        // TODO(ngates): maybe we should deprecate either this or IntoArrayData.
        array.into_array()
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
                fn metadata(&self) -> [<$Name Metadata>] {
                    use $crate::metadata::TryDeserializeArrayMetadata;
                    use vortex_error::VortexExpect;
                    [<$Name Metadata>]::try_deserialize_metadata(Some(
                        self.as_ref().metadata()
                            .vortex_expect("FIXME(ngates): OwnedData to hold metadata bytes")
                            .as_ref(),
                    )).vortex_expect("Metadata validated on creation of ArrayData")
                }

                // fn metadata(&self) -> Box<dyn $crate::ArrayMetadata> {
                //     self.as_ref().encoding().metadata(self.as_ref())
                //         .vortex_expect("Metadata validated on creation of ArrayData")
                // }

                pub fn len(&self) -> usize {
                    self.as_ref().len()
                }

                pub fn is_empty(&self) -> bool {
                    self.as_ref().is_empty()
                }

                #[allow(dead_code)]
                fn try_from_parts(
                    dtype: vortex_dtype::DType,
                    len: usize,
                    metadata: [<$Name Metadata>],
                    children: std::sync::Arc<[$crate::ArrayData]>,
                    stats: $crate::stats::StatsSet,
                ) -> VortexResult<Self> {
                    Self::try_from($crate::ArrayData::try_new_owned(
                            &[<$Name Encoding>],
                            dtype,
                            len,
                            std::sync::Arc::new(metadata),
                            None,
                            children,
                            stats
                    )?)
                }
            }

            impl TryFrom<$crate::ArrayData> for [<$Name Array>] {
                type Error = vortex_error::VortexError;

                fn try_from(value: $crate::ArrayData) -> Result<Self, Self::Error> {
                    if value.encoding().id() != <$Name as $crate::ArrayDef>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            value.encoding().id().as_ref(),
                            <$Name as $crate::ArrayDef>::ID,
                        );
                    }
                    // SAFETY: We know that our Array struct has an identical layout to ArrayData.
                    Ok(unsafe { std::mem::transmute(value) })
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
            impl $crate::encoding::ArrayMetadataVTable<$crate::ArrayData> for [<$Name Encoding>] {
                fn metadata(&self, array: &$crate::ArrayData) -> vortex_error::VortexResult<Box<dyn $crate::ArrayMetadata>> {
                    use $crate::metadata::TryDeserializeArrayMetadata;
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
