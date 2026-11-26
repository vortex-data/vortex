// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A structured, shrinkable description of a `file_io` case for ephyra.
//!
//! Like `ArrayOpsPlan`, a [`FileIoPlan`] is plain data that ephyra shrinks field by field, and it
//! materializes into the same [`FuzzFileAction`] the libFuzzer target checks.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use ephyra::gens;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_expr::arbitrary::filter_expr;
use vortex_expr::arbitrary::projection_expr;

use super::FuzzFileAction;
use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::array::CompressorStrategy;

/// An array to write, the expressions to scan it back with, and the write strategy.
#[derive(Clone, Debug, ephyra::Arbitrary, serde::Serialize, serde::Deserialize)]
pub struct FileIoPlan {
    /// The longest array, usually short so edge cases near empty files come up often.
    #[ephyra(gen = gens::frequency(vec![
        (3, gens::int(0u16..=64).boxed()),
        (1, gens::int(0u16..=u16::try_from(FUZZ_FILE_ARRAY_MAX_LEN).unwrap_or(u16::MAX)).boxed()),
    ]))]
    pub max_len: u16,
    /// Bytes for the fields of a non-nullable top-level struct; `None` lets the array bytes pick
    /// any dtype. Projections and filters only apply to structs, which random dtypes rarely are.
    #[ephyra(gen = gens::option(gens::vec(gens::int(0u8..=255)).len(0..=64)))]
    pub struct_fields: Option<Vec<u8>>,
    /// Bytes decoded into the array.
    #[ephyra(gen = gens::vec(gens::int(0u8..=255)).len(0..=4096))]
    pub array_bytes: Vec<u8>,
    /// Bytes for the projection expression; `None` reads every column.
    #[ephyra(gen = gens::option(gens::vec(gens::int(0u8..=255)).len(0..=64)))]
    pub projection: Option<Vec<u8>>,
    /// Bytes for the filter expression; `None` keeps every row.
    #[ephyra(gen = gens::option(gens::vec(gens::int(0u8..=255)).len(0..=64)))]
    pub filter: Option<Vec<u8>>,
    /// Write with the compact BtrBlocks strategy instead of the default.
    pub compact: bool,
}

impl FileIoPlan {
    /// Decode the plan into the libFuzzer target's input, or `None` when its bytes do not decode.
    pub fn materialize(&self) -> Option<FuzzFileAction> {
        let dtype = match &self.struct_fields {
            Some(bytes) => Some(DType::Struct(
                StructFields::arbitrary(&mut Unstructured::new(bytes)).ok()?,
                Nullability::NonNullable,
            )),
            None => None,
        };
        let array = ArbitraryArray::arbitrary_with_config(
            &mut Unstructured::new(&self.array_bytes),
            &ArbitraryArrayConfig {
                dtype,
                len: 0..=usize::from(self.max_len).min(FUZZ_FILE_ARRAY_MAX_LEN),
            },
        )
        .ok()?
        .0;
        let dtype = array.dtype().clone();
        let expression = |bytes: &Option<Vec<u8>>, build: ExprBuilder| {
            bytes
                .as_ref()
                .and_then(|bytes| build(&mut Unstructured::new(bytes), &dtype).ok().flatten())
        };
        Some(FuzzFileAction {
            projection_expr: expression(&self.projection, projection_expr),
            filter_expr: expression(&self.filter, filter_expr),
            compressor_strategy: if self.compact {
                compact_strategy()
            } else {
                CompressorStrategy::Default
            },
            array,
        })
    }
}

type ExprBuilder =
    fn(&mut Unstructured<'_>, &DType) -> arbitrary::Result<Option<vortex::expr::Expression>>;

#[cfg(feature = "zstd")]
fn compact_strategy() -> CompressorStrategy {
    CompressorStrategy::Compact
}

#[cfg(not(feature = "zstd"))]
fn compact_strategy() -> CompressorStrategy {
    CompressorStrategy::Default
}
