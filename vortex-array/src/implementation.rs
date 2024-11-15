//! The core Vortex macro to create new encodings and array types.

use vortex_buffer::Buffer;
use vortex_error::{vortex_bail, VortexError, VortexExpect as _, VortexResult};

use crate::array::visitor::ArrayVisitor;
use crate::encoding::{ArrayEncoding, ArrayEncodingExt, ArrayEncodingRef, EncodingId, EncodingRef};
use crate::stats::ArrayStatistics;
use crate::{
    ArrayDType, ArrayData, ArrayMetadata, ArrayTrait, GetArrayMetadata, InnerArrayData,
    IntoArrayData, OwnedArrayData, ToOwnedArrayData, TryDeserializeArrayMetadata,
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

            #[derive(std::fmt::Debug, Clone)]
            pub struct [<$Name Array>] {
                typed: $crate::TypedArray<$Name>
            }
            impl AsRef<$crate::ArrayData> for [<$Name Array>] {
                fn as_ref(&self) -> &$crate::ArrayData {
                    self.typed.array()
                }
            }
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

impl<D> ToOwnedArrayData for D
where
    D: IntoArrayData + ArrayEncodingRef + ArrayStatistics + GetArrayMetadata + Clone,
{
    fn to_owned_array_data(&self) -> OwnedArrayData {
        let array = self.clone().into_array();
        match array.0 {
            InnerArrayData::Owned(d) => d,
            InnerArrayData::Viewed(ref view) => {
                struct Visitor {
                    buffer: Option<Buffer>,
                    children: Vec<ArrayData>,
                }
                impl ArrayVisitor for Visitor {
                    fn visit_child(&mut self, _name: &str, array: &ArrayData) -> VortexResult<()> {
                        self.children.push(array.clone());
                        Ok(())
                    }

                    fn visit_buffer(&mut self, buffer: &Buffer) -> VortexResult<()> {
                        if self.buffer.is_some() {
                            vortex_bail!("Multiple buffers found in view")
                        }
                        self.buffer = Some(buffer.clone());
                        Ok(())
                    }
                }
                let mut visitor = Visitor {
                    buffer: None,
                    children: vec![],
                };
                array.with_dyn(|a| {
                    a.accept(&mut visitor)
                        .vortex_expect("Error while visiting Array View children")
                });
                OwnedArrayData::try_new(
                    view.encoding(),
                    array.dtype().clone(),
                    array.len(),
                    self.metadata(),
                    visitor.buffer,
                    visitor.children.into(),
                    view.statistics().to_set(),
                )
                .vortex_expect("Failed to create ArrayData from Array View")
            }
        }
    }
}

impl AsRef<ArrayData> for ArrayData {
    fn as_ref(&self) -> &ArrayData {
        self
    }
}
