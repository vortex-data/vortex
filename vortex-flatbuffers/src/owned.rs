use vortex_error::VortexResult;

use crate::FlatBuffer;

pub trait Owned {
    fn try_new(buffer: FlatBuffer) -> VortexResult<Self>
    where
        Self: Sized,
    {
        Self::try_new_with_loc(buffer, 0)
    }

    fn try_new_with_loc(buffer: FlatBuffer, loc: usize) -> VortexResult<Self>
    where
        Self: Sized;
}

#[allow(unused_macros)]
macro_rules! make_owned {
    ($name:ident) => {
        paste::paste! {
            #[derive(Clone)]
            pub struct [<Owned $name>]($crate::FlatBuffer);

            impl $crate::owned::Owned for [<Owned $name>] {
                /// Attempt to construct a new owned type that wraps a `flatbuffers` type.
                fn try_new_with_loc(buffer: $crate::FlatBuffer, loc: usize) -> vortex_error::VortexResult<Self> {
                    // Perform validation when we move into the OwnedTable.
                    let opts = flatbuffers::VerifierOptions::default();
                    let mut verifier = flatbuffers::Verifier::new(&opts, &buffer);
                    <flatbuffers::ForwardsUOffset<$name> as flatbuffers::Verifiable>::run_verifier(&mut verifier, loc)?;

                    Ok(Self(buffer))
                }
            }

            impl [<Owned $name>] {

                /// Create a new buffer directly.
                ///
                /// # Safety
                ///
                /// It is the caller's responsibility to be absolutely certain that the provided buffer is directly
                /// populated with a valid Flatbuffer bytes of the desired type.
                pub unsafe fn new_unchecked(buffer: $crate::FlatBuffer) -> Self {
                    Self(buffer)
                }

                /// Access the `flatbuffers` inner type without verification.
                ///
                /// This is safe because verification is run on construction.
                pub fn as_fb(&self) -> $name {
                    // SAFETY: we run verification on construction.
                    unsafe { flatbuffers::root_unchecked::<$name>(&self.0) }
                }

                /// Consume the owned reference, handing back the underlying buffer.
                pub fn into_inner(self) -> $crate::FlatBuffer {
                    self.0
                }

                /// Create a new owned child of type `T` over the provided slice range.
                ///
                /// # Panics
                ///
                /// If the provided slice is not a valid range within the owned buffer,
                /// the program will panic.
                pub fn owned_child<'a, T, O>(&self, slice: &'a [u8]) -> vortex_error::VortexResult<O>
                where
                    T: flatbuffers::Follow<'a>,
                    O: $crate::owned::Owned + 'static,
                {
                    let Some(range) = self.0.subslice_range(slice) else {
                        vortex_error::vortex_panic!("provided slice is not valid subrange of {}", stringify!([<Owned $name>]))
                    };

                    // Create a new buffer with the provided offset.
                    O::try_new_with_loc(self.0.clone(), range.start)
                }
            }
        }
    };
}

#[cfg(feature = "array")]
pub mod array {
    use crate::array::Array;

    make_owned!(Array);
}

/// Owned versions of the Flatbuffer types defined in the `message` package.
#[cfg(feature = "ipc")]
pub mod message {
    use crate::message::Message;

    make_owned!(Message);
}

/// Owned versions of the Flatbuffer types defined in the `layout` package.
#[cfg(feature = "layout")]
pub mod layout {
    use crate::layout::Layout;

    make_owned!(Layout);
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use vortex_buffer::{Alignment, Buffer};

    use super::layout::OwnedLayout;
    use crate::layout::{Layout, LayoutArgs};
    use crate::owned::Owned;
    use crate::FlatBuffer;

    #[test]
    fn test_owned_buffer() {
        let mut fbb = FlatBufferBuilder::new();
        let offset = Layout::create(
            &mut fbb,
            &LayoutArgs {
                encoding: 1,
                row_count: 1,
                ..Default::default()
            },
        );

        fbb.finish_minimal(offset);

        let (buffer, start) = fbb.collapse();
        let buffer = Buffer::copy_from_aligned(&buffer[start..buffer.len()], Alignment::new(8));
        let buffer = FlatBuffer::try_from(buffer).unwrap();

        // We can't refer to an unowned type
        let owned = OwnedLayout::try_new(buffer).unwrap();
        assert_eq!(owned.as_fb().encoding(), 1);
        assert_eq!(owned.as_fb().row_count(), 1);
    }
}
