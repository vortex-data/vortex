//! The core Vortex macro to create new encodings and array types.

use vortex_error::VortexError;

use crate::encoding::{
    ArrayEncodingExt, ArrayEncodingRef, EncodingId, EncodingRef, EncodingVTable,
};
use crate::{ArrayData, ArrayMetadata, ArrayTrait, ToArrayData, TryDeserializeArrayMetadata};

/// Trait the defines the set of types relating to an array.
/// Because it has associated types it can't be used as a trait object.
pub trait ArrayDef {
    const ID: EncodingId;
    const ENCODING: EncodingRef;

    type Array: ArrayTrait + TryFrom<ArrayData, Error = VortexError>;
    type Metadata: ArrayMetadata + Clone + for<'m> TryDeserializeArrayMetadata<'m>;
    type Encoding: EncodingVTable + ArrayEncodingExt<D = Self>;
}

impl<A: AsRef<ArrayData>> ToArrayData for A {
    fn to_array(&self) -> ArrayData {
        self.as_ref().clone()
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

            impl $crate::encoding::Encoding for [<$Name Encoding>] {
                type Array = [<$Name Array>];
                type Metadata = [<$Name Metadata>];
            }

            #[derive(std::fmt::Debug, Clone)]
            #[repr(transparent)]
            pub struct [<$Name Array>]($crate::ArrayData);

            impl $crate::IntoArrayData for [<$Name Array>] {
                fn into_array(self) -> $crate::ArrayData {
                    self.0
                }
            }
            impl AsRef<$crate::ArrayData> for [<$Name Array>] {
                fn as_ref(&self) -> &$crate::ArrayData {
                    &self.0
                }
            }

            impl [<$Name Array>] {
                #[allow(dead_code)]
                fn metadata(&self) -> &[<$Name Metadata>] {
                    use vortex_error::VortexExpect;
                    self.0.metadata::<[<$Name Metadata>]>()
                        .vortex_expect("Metadata should be tied to the encoding")
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

                fn try_from(data: $crate::ArrayData) -> vortex_error::VortexResult<Self> {
                    if data.encoding().id() != <$Name as $crate::ArrayDef>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().id().as_ref(),
                            <$Name as $crate::ArrayDef>::ID,
                        );
                    }
                    Ok(Self(data))
                }
            }

            // NOTE(ngates): this is the cheeky one.... Since we know that Arrays are structurally
            //  equal to ArrayData, we can transmute a &ArrayData to a &Array.
            impl<'a> TryFrom<&'a $crate::ArrayData> for &'a [<$Name Array>] {
                type Error = vortex_error::VortexError;

                fn try_from(data: &'a $crate::ArrayData) -> vortex_error::VortexResult<Self> {
                    if data.encoding().id() != <$Name as $crate::ArrayDef>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().id().as_ref(),
                            <$Name as $crate::ArrayDef>::ID,
                        );
                    }
                    Ok(unsafe { std::mem::transmute::<&$crate::ArrayData, &[<$Name Array>]>(data) })
                }
            }

            /// The array encoding
            #[derive(std::fmt::Debug)]
            pub struct [<$Name Encoding>];
            impl $crate::encoding::EncodingVTable for [<$Name Encoding>] {
                #[inline]
                fn id(&self) -> $crate::encoding::EncodingId {
                    <$Name as $crate::ArrayDef>::ID
                }

                fn as_any(&self) -> &dyn std::any::Any {
                    self
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
