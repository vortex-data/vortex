//! The core Vortex macro to create new encodings and array types.

use std::fmt::Display;

use crate::ArrayData;
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

            #[allow(dead_code)]
            impl [<$Name Array>] {
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
                            [<$Name Encoding>]::vtable(),
                            dtype,
                            len,
                            metadata.serialize()?,
                            buffers,
                            children,
                            stats
                    )?)
                }

                fn metadata(&self) -> <$Metadata as $crate::DeserializeMetadata>::Output {
                    use $crate::DeserializeMetadata;

                    // SAFETY: Metadata is validated during construction of ArrayData.
                    unsafe { <$Metadata as DeserializeMetadata>::deserialize_unchecked(self.0.metadata_bytes()) }
                }

                /// Optionally downcast an [`ArrayData`](crate::ArrayData) instance to a specific encoding.
                ///
                /// Preferred in cases where a backtrace isn't needed, like when trying multiple encoding to go
                /// down different code paths.
                pub fn maybe_from(data: impl AsRef<$crate::ArrayData>) -> Option<Self> {
                    let data = data.as_ref();
                    (data.encoding() == <[<$Name Encoding>] as $crate::Encoding>::ID).then_some(Self(data.clone()))
                }
            }

            impl std::ops::Deref for [<$Name Array>] {
                type Target = $crate::ArrayData;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl TryFrom<$crate::ArrayData> for [<$Name Array>] {
                type Error = vortex_error::VortexError;

                fn try_from(data: $crate::ArrayData) -> vortex_error::VortexResult<Self> {
                    if data.encoding() != <[<$Name Encoding>] as $crate::Encoding>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().as_ref(),
                            <[<$Name Encoding>] as $crate::Encoding>::ID,
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
                    if data.encoding() != <[<$Name Encoding>] as $crate::Encoding>::ID {
                        vortex_error::vortex_bail!(
                            "Mismatched encoding {}, expected {}",
                            data.encoding().as_ref(),
                            <[<$Name Encoding>] as $crate::Encoding>::ID,
                        );
                    }
                    Ok(unsafe { std::mem::transmute::<&$crate::ArrayData, &[<$Name Array>]>(data) })
                }
            }

            /// The array encoding
            #[derive(std::fmt::Debug)]
            pub struct [<$Name Encoding>];

            impl [<$Name Encoding>] {
                pub const fn vtable() -> $crate::vtable::VTableRef {
                    $crate::vtable::VTableRef::from_static(&Self)
                }
            }

            impl $crate::Encoding for [<$Name Encoding>] {
                const ID: $crate::EncodingId = $crate::EncodingId::new($id, $code);
                type Array = [<$Name Array>];
                type Metadata = $Metadata;
            }

            impl $crate::vtable::EncodingVTable for [<$Name Encoding>] {
                #[inline]
                fn id(&self) -> $crate::EncodingId {
                    <[<$Name Encoding>] as $crate::Encoding>::ID
                }

                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }
            }
        }
    };
}

impl AsRef<ArrayData> for ArrayData {
    fn as_ref(&self) -> &ArrayData {
        self
    }
}
