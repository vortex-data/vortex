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
                    unsafe { &*(self as *const [<$V VTable>] as *const $crate::ExtensionTypeRef<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V VTable>] {
                type Target = dyn $crate::ExtensionType;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an LayoutAdapter.
                    unsafe { &*(self as *const [<$V VTable>] as *const $crate::ExtensionTypeAdapter<[<$V VTable>]>) }
                }
            }

            impl $crate::IntoExtensionTypeRef for [<$V VTable>] {
                fn into_extension_type_ref(self) -> $crate::ExtensionTypeRef {
                    // We can unsafe transmute ourselves to an LayoutAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V VTable>], $crate::ExtensionTypeAdapter::<[<$V VTable>]>>(self) })
                }
            }
        }
    };
}
