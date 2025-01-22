//! The core Vortex macro to create new encodings and array types.

use std::fmt::{Display, Formatter};

use crate::array::StructMetadata;
use crate::encoding::{ArrayEncodingRef, EncodingRef};
use crate::{ArrayData, ToArrayData};

impl<A: AsRef<ArrayData>> ToArrayData for A {
    fn to_array(&self) -> ArrayData {
        self.as_ref().clone()
    }
}

/// Macro to generate all the necessary code for a new type of array encoding. Including:
/// 1. New Array type that implements `AsRef<ArrayData>`, `GetArrayMetadata`, `ToArray`, `IntoArray`, and multiple useful `From`/`TryFrom` implementations.
/// 2. New Encoding type that implements `ArrayEncoding`.
/// 3. New metadata type that implements `ArrayMetadata`.
#[macro_export]
macro_rules! impl_encoding {
    ($id:literal, $code:expr, $Name:ident, $Metadata:ty) => {
        $crate::paste::paste! {
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
                fn try_from_parts(
                    dtype: vortex_dtype::DType,
                    len: usize,
                    metadata: $Metadata,
                    buffers: Option<Box<[vortex_buffer::ByteBuffer]>>,
                    children: Option<Box<[$crate::ArrayData]>>,
                    stats: $crate::stats::StatsSet,
                ) -> VortexResult<Self> {
                    use $crate::SerializeMetadata;

                    Self::try_from($crate::ArrayData::try_new_owned(
                            &[<$Name Encoding>],
                            dtype,
                            len,
                            metadata.serialize()?,
                            buffers,
                            children,
                            stats
                    )?)
                }

                /// Optionally downcast an [`ArrayData`](crate::ArrayData) instance to a specific encoding.
                ///
                /// Preferred in cases where a backtrace isn't needed, like when trying multiple encoding to go
                /// down different code paths.
                pub fn maybe_from(data: impl AsRef<$crate::ArrayData>) -> Option<Self> {
                    let data = data.as_ref();
                    (data.encoding().id() == <[<$Name Encoding>] as $crate::encoding::Encoding>::ID).then_some(Self(data.clone()))
                }
            }

            impl $crate::ArrayTrait for [<$Name Array>] {}

            impl TryFrom<$crate::ArrayData> for [<$Name Array>] {
                type Error = vortex_error::VortexError;

                fn try_from(data: $crate::ArrayData) -> vortex_error::VortexResult<Self> {
                    if data.encoding().id() != <[<$Name Encoding>] as $crate::encoding::Encoding>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().id().as_ref(),
                            <[<$Name Encoding>] as $crate::encoding::Encoding>::ID,
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
                    if data.encoding().id() != <[<$Name Encoding>] as $crate::encoding::Encoding>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().id().as_ref(),
                            <[<$Name Encoding>] as $crate::encoding::Encoding>::ID,
                        );
                    }
                    Ok(unsafe { std::mem::transmute::<&$crate::ArrayData, &[<$Name Array>]>(data) })
                }
            }

            /// The array encoding
            #[derive(std::fmt::Debug)]
            pub struct [<$Name Encoding>];

            impl $crate::encoding::Encoding for [<$Name Encoding>] {
                const ID: $crate::encoding::EncodingId = $crate::encoding::EncodingId::new($id, $code);
                type Array = [<$Name Array>];
                type Metadata = $Metadata;
            }

            impl $crate::encoding::EncodingVTable for [<$Name Encoding>] {
                #[inline]
                fn id(&self) -> $crate::encoding::EncodingId {
                    <[<$Name Encoding>] as $crate::encoding::Encoding>::ID
                }

                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }

                fn metadata_display(&self, array: &$crate::ArrayData, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    use std::fmt::Display;
                    use $crate::DeserializeMetadata;

                    match <$Metadata as DeserializeMetadata>::deserialize(array.metadata_bytes()) {
                        Ok(metadata) => metadata.fmt(f),
                        Err(_) => write!(f, "Error deserializing metadata"),
                    }
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
