/// Create a new extension type VTable implementation.
#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct [<$V VTable>];

            impl AsRef<dyn $crate::ExtensionType> for [<$V VTable>] {
                fn as_ref(&self) -> &dyn $crate::ExtensionType {
                    // We can unsafe cast ourselves to a LayoutAdapter.
                    unsafe { &*(self as *const [<$V VTable>] as *const $crate::ExtensionTypeAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V ExtensionType>] {
                type Target = dyn $crate::ExtensionType;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an adapter.
                    unsafe { &*(self as *const [<$V ExtensionType>] as *const $crate::ExtensionTypeAdapter<[<$V VTable>]>) }
                }
            }

            impl $crate::IntoExtensionTypeRef for [<$V ExtensionType>] {
                fn into_extension_type_ref(self) -> $crate::ExtensionTypeRef {
                    // We can transmute ourselves to an ExtensionTypeAdapter because
                    // it is the only implementation of the sealed trait.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V ExtensionType>], $crate::ExtensionTypeAdapter::<[<$V VTable>]>>(self) })
                }
            }

            impl AsRef<dyn $crate::ExtensionTypeEncoding> for [<$V ExtensionTypeEncoding>] {
                fn as_ref(&self) -> &dyn $crate::ExtensionTypeEncoding {
                    unsafe { &*(self as *const [<$V ExtensionTypeEncoding>] as *const $crate::ExtensionTypeEncodingAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V ExtensionTypeEncoding>] {
                type Target = dyn $crate::ExtensionTypeEncoding;

                fn deref(&self) -> &Self::Target {
                    // It is safe to pointer cast to the adapter type, because we know it is the
                    // only implementation of the sealed trait.
                    unsafe { &*(self as *const [<$V ExtensionTypeEncoding>] as *const $crate::ExtensionTypeEncodingAdapter<[<$V VTable>]>) }
                }
            }
        }
    };
}
