// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BtrBlocks-specific compressor wrapping the generic [`CascadingCompressor`].

use std::ops::Deref;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::BtrBlocksCompressorBuilder;
use crate::CascadingCompressor;

/// The BtrBlocks-style compressor with all built-in schemes pre-registered.
///
/// This is a thin wrapper around [`CascadingCompressor`] that provides a default set of
/// compression schemes via [`BtrBlocksCompressorBuilder`].
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor - all schemes allowed.
/// let compressor = BtrBlocksCompressor::default();
///
/// // Exclude specific schemes using the builder.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor(
    /// The underlying cascading compressor.
    pub CascadingCompressor,
);

impl BtrBlocksCompressor {
    /// Compresses an array using BtrBlocks-inspired compression.
    pub fn compress(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        self.0.compress(array)
    }
}

impl Deref for BtrBlocksCompressor {
    type Target = CascadingCompressor;

    fn deref(&self) -> &CascadingCompressor {
        &self.0
    }
}

impl Default for BtrBlocksCompressor {
    fn default() -> Self {
        BtrBlocksCompressorBuilder::default().build()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::List;
    use vortex_array::arrays::ListView;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::BtrBlocksCompressor;

    #[rstest]
    #[case::zctl(
        unsafe {
            ListViewArray::new_unchecked(
                buffer![1i32, 2, 3, 4, 5].into_array(),
                buffer![0i32, 3].into_array(),
                buffer![3i32, 2].into_array(),
                Validity::NonNullable,
            ).with_zero_copy_to_list(true)
        },
        true,
    )]
    #[case::overlapping(
        ListViewArray::new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0i32, 0, 0].into_array(),
            buffer![3i32, 3, 3].into_array(),
            Validity::NonNullable,
        ),
        false,
    )]
    fn listview_compress_roundtrip(
        #[case] input: ListViewArray,
        #[case] expect_list: bool,
    ) -> VortexResult<()> {
        let array_ref = input.clone().into_array();
        let result = BtrBlocksCompressor::default().compress(&array_ref)?;
        if expect_list {
            assert!(result.as_opt::<List>().is_some());
        } else {
            assert!(result.as_opt::<ListView>().is_some());
        }
        assert_arrays_eq!(result, input);
        Ok(())
    }

    #[test]
    fn test_constant_all_true() -> VortexResult<()> {
        let array = BoolArray::new(BitBuffer::from(vec![true; 100]), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.clone().into_array())?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array);
        Ok(())
    }

    #[test]
    fn test_constant_all_false() -> VortexResult<()> {
        let array = BoolArray::new(BitBuffer::from(vec![false; 100]), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.clone().into_array())?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array);
        Ok(())
    }

    #[test]
    fn test_nullable_all_valid_compressed() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![true; 100]),
            Validity::from(BitBuffer::from(vec![true; 100])),
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.clone().into_array())?;
        assert!(compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array);
        Ok(())
    }

    #[test]
    fn test_nullable_with_nulls_not_compressed() -> VortexResult<()> {
        let validity = Validity::from(BitBuffer::from_iter((0..100).map(|i| i % 3 != 0)));
        let array = BoolArray::new(BitBuffer::from(vec![true; 100]), validity);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.clone().into_array())?;
        assert!(!compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array);
        Ok(())
    }

    #[test]
    fn test_mixed_not_constant() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true, false, true]),
            Validity::NonNullable,
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.clone().into_array())?;
        assert!(!compressed.is::<Constant>());
        assert_arrays_eq!(compressed, array);
        Ok(())
    }
}
