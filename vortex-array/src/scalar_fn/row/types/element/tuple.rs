// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`]s, and the per-argument decode behind them.

use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::LaneZip;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Extension;
use crate::arrays::Masked;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::InputElement;

mod private {
    pub trait Sealed {}
}

/// One decoded input, collapsed to a single row when it is constant for the batch.
pub struct ArgColumn<T: InputElement>(
    /// The decoded column, classified by whether it varies within the batch.
    ArgColumnKind<T>,
);

enum ArgColumnKind<T: InputElement> {
    Varying(T::Column),
    Constant(T::Column),
}

impl<T: InputElement> ArgColumn<T> {
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // An empty input has no row 0 to slice, and its row loop runs zero times either way.
        if let Some(constant) = batch_constant(&array)
            && !array.is_empty()
        {
            return Ok(Self(ArgColumnKind::Constant(T::decode(
                constant.slice(0..1)?,
                ctx,
            )?)));
        }

        Ok(Self(ArgColumnKind::Varying(T::decode(array, ctx)?)))
    }

    fn decode_null_tolerant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        // Batch execution short-circuits null constants before selecting a strategy, so a
        // constant reaching this path is non-null and can use the ordinary decode.
        if let Some(constant) = batch_constant(&array)
            && !array.is_empty()
        {
            return Ok(Some(Self(ArgColumnKind::Constant(T::decode(
                constant.slice(0..1)?,
                ctx,
            )?))));
        }

        Ok(T::decode_null_tolerant(array, ctx)?
            .map(ArgColumnKind::Varying)
            .map(Self))
    }

    fn get(&self, index: usize) -> T::Elem<'_> {
        match &self.0 {
            ArgColumnKind::Varying(column) => T::get(column, index),
            ArgColumnKind::Constant(column) => T::get(column, 0),
        }
    }

    fn varying(&self) -> Option<&T::Column> {
        match &self.0 {
            ArgColumnKind::Varying(column) => Some(column),
            ArgColumnKind::Constant(_) => None,
        }
    }

    fn addresses_rows(&self, row_count: usize) -> bool {
        // A constant is always read at index zero, so it addresses any batch length.
        match &self.0 {
            ArgColumnKind::Varying(column) => T::varying_len(&T::varying(column)) == row_count,
            ArgColumnKind::Constant(_) => true,
        }
    }

    fn constant(&self) -> Option<T::Elem<'_>> {
        match &self.0 {
            ArgColumnKind::Varying(_) => None,
            ArgColumnKind::Constant(column) => Some(T::get(column, 0)),
        }
    }
}

/// Return the batch-constant array, looking through masked and extension wrappers.
///
/// Batch execution owns mask validity, so a masked constant may expose its constant child here. An
/// extension over constant storage remains wrapped to preserve its extension dtype.
pub fn batch_constant(array: &ArrayRef) -> Option<ArrayRef> {
    if array.as_constant().is_some() {
        return Some(array.clone());
    }

    if let Some(masked) = array.as_opt::<Masked>() {
        return Some(masked.child().clone()).filter(|child| child.as_constant().is_some());
    }

    array
        .as_opt::<Extension>()
        .is_some_and(|ext| ext.storage_array().as_constant().is_some())
        .then(|| array.clone())
}

/// Typed argument tuples for arities zero through twelve.
///
/// This trait is sealed; add a new row representation by implementing [`InputElement`] and placing
/// it in one of the supplied tuples.
pub trait ElementTuple: 'static + private::Sealed {
    /// The decoded column representations.
    type Columns;

    /// Direct references to decoded columns when every argument varies within the batch.
    type VaryingColumns<'a>;

    /// The borrowed row of element values.
    type Elems<'a>;

    /// The batch-constant element values: [`Elems`](Self::Elems) with every argument wrapped in
    /// `Option`.
    ///
    /// `Some` marks an argument whose operand is constant for the batch and carries the element
    /// every row reads; `None` marks one that varies by row. This is what
    /// [`RowVisitor`](crate::scalar_fn::RowVisitor) hands to a visit's prepare closure, so a kernel
    /// can hoist work that depends only on a constant argument out of the row loop.
    type ConstElems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether _any_ argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// Validate the input dtypes, including that `dtypes` has exactly `ARITY` entries.
    ///
    /// The expression layer checks the count against [`Arity`](crate::scalar_fn::Arity) before it
    /// builds a call, but this is also the entry point of the public
    /// [`return_dtype`](crate::scalar_fn::ScalarFnVTable::return_dtype), so the count is enforced
    /// here rather than assumed.
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once. Called once per batch.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Decode every input column once while tolerating null rows.
    ///
    /// Return `Ok(None)` when an argument has no null-tolerant representation. The skip-invalid
    /// strategy calls this once per batch.
    fn decode_null_tolerant(
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Columns>>;

    /// Read the row of elements at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_>;

    /// Borrow every decoded column directly, or `None` when any argument is batch-constant.
    ///
    /// This is selected once outside the hot loop. Keeping `ArgColumn` out of the resulting tuple
    /// gives the optimizer ordinary contiguous column access without a per-row constant check.
    fn varying(columns: &Self::Columns) -> Option<Self::VaryingColumns<'_>>;

    /// Whether every varying column contains exactly `row_count` rows.
    fn varying_len_matches(columns: &Self::VaryingColumns<'_>, row_count: usize) -> bool;

    /// Whether every argument that varies within the batch contains exactly `row_count` rows.
    ///
    /// The same guarantee as [`varying_len_matches`](Self::varying_len_matches), for the mixed case
    /// [`varying`](Self::varying) declines: a batch-constant argument is exempt because it was
    /// collapsed to one row, while every argument beside it still has to address the whole batch.
    fn decoded_lens_match(columns: &Self::Columns, row_count: usize) -> bool;

    /// Read one row from columns already known to vary within the batch.
    fn get_varying<'a>(columns: &Self::VaryingColumns<'a>, index: usize) -> Self::Elems<'a>;

    /// Read one row from varying columns without checking bounds.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds for every column.
    unsafe fn get_varying_unchecked<'a>(
        columns: &Self::VaryingColumns<'a>,
        index: usize,
    ) -> Self::Elems<'a>;

    /// Read the batch-constant elements out of the decoded columns. Called once per batch.
    fn constants(columns: &Self::Columns) -> Self::ConstElems<'_>;
}

/// An argument tuple that supports a validated dense indexed traversal.
///
/// This is separate from [`ElementTuple`] because many row elements have no contiguous source, and
/// stable Rust cannot provide a blanket fallback plus a more specific primitive implementation.
/// The trait is sealed so shared execution can rely on its unchecked-read contract. A tuple only
/// implements it when the source can be validated once and every lane can then be read
/// independently.
pub trait IndexedElementTuple: ElementTuple {
    /// The source shared execution uses for a dense all-varying loop.
    ///
    /// Its length must be the common varying-column length. For every valid index it must preserve
    /// row order, return the same value as [`ElementTuple::get_varying`], and uphold the unchecked
    /// read contract of [`IndexedSource`].
    type Source<'a>: IndexedSource<Item = Self::Elems<'a>>;

    /// Borrow a source from columns already validated to vary within the batch.
    fn indexed_source<'a>(columns: &Self::VaryingColumns<'a>) -> Self::Source<'a>;
}

/// An indexed native slice yielding the one-tuples expected by a unary row closure.
#[derive(Clone, Copy)]
pub struct UnaryTupleSource<'a, T>(
    /// The native values read by the row loop.
    &'a [T],
);

impl<T: Copy> IndexedSource for UnaryTupleSource<'_, T> {
    type Item = (T,);

    fn len(&self) -> usize {
        self.0.len()
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        // SAFETY: the caller guarantees that `index` is in bounds.
        (unsafe { *self.0.get_unchecked(index) },)
    }
}

impl private::Sealed for () {}

impl ElementTuple for () {
    type Columns = ();
    type VaryingColumns<'a> = ();
    type Elems<'a> = ();
    type ConstElems<'a> = ();

    const ARITY: usize = 0;
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtypes: &[DType]) -> VortexResult<()> {
        vortex_ensure_eq!(
            dtypes.len(),
            0,
            "expected 0 argument dtypes, got {}",
            dtypes.len(),
        );
        Ok(())
    }

    fn decode(_args: &dyn ExecutionArgs, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns> {
        Ok(())
    }

    fn decode_null_tolerant(
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Columns>> {
        Ok(Some(()))
    }

    fn get(_columns: &Self::Columns, _index: usize) -> Self::Elems<'_> {}

    fn varying(_columns: &Self::Columns) -> Option<Self::VaryingColumns<'_>> {
        Some(())
    }

    fn varying_len_matches(_columns: &Self::VaryingColumns<'_>, _row_count: usize) -> bool {
        true
    }

    fn decoded_lens_match(_columns: &Self::Columns, _row_count: usize) -> bool {
        true
    }

    fn get_varying<'a>(_columns: &Self::VaryingColumns<'a>, _index: usize) -> Self::Elems<'a> {}

    unsafe fn get_varying_unchecked<'a>(
        _columns: &Self::VaryingColumns<'a>,
        _index: usize,
    ) -> Self::Elems<'a> {
    }

    fn constants(_columns: &Self::Columns) -> Self::ConstElems<'_> {}
}

macro_rules! element_tuple {
    ($arity:literal; $($t:ident : $idx:tt),+) => {
        impl<$($t: InputElement),+> private::Sealed for ($($t,)+) {}

        impl<$($t: InputElement),+> ElementTuple for ($($t,)+) {
            type Columns = ($(ArgColumn<$t>,)+);
            type VaryingColumns<'a> = ($($t::Varying<'a>,)+);
            type Elems<'a> = ($($t::Elem<'a>,)+);
            type ConstElems<'a> = ($(Option<$t::Elem<'a>>,)+);

            const ARITY: usize = $arity;
            const DENSE_SAFE: bool = $($t::DENSE_SAFE &&)+ true;
            const DECODE_FALLIBLE: bool = $($t::DECODE_FALLIBLE ||)+ false;

            fn validate(dtypes: &[DType]) -> VortexResult<()> {
                vortex_ensure_eq!(
                    dtypes.len(),
                    $arity,
                    "expected {} argument dtypes, got {}",
                    $arity,
                    dtypes.len(),
                );

                $($t::validate(&dtypes[$idx])?;)+
                Ok(())
            }

            fn decode(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Self::Columns> {
                Ok(($(ArgColumn::<$t>::decode(args.get($idx)?, ctx)?,)+))
            }

            fn decode_null_tolerant(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<Self::Columns>> {
                Ok(Some((
                    $(match ArgColumn::<$t>::decode_null_tolerant(args.get($idx)?, ctx)? {
                        Some(column) => column,
                        None => return Ok(None),
                    },)+
                )))
            }

            fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_> {
                ($(columns.$idx.get(index),)+)
            }

            fn varying(columns: &Self::Columns) -> Option<Self::VaryingColumns<'_>> {
                Some(($($t::varying(columns.$idx.varying()?),)+))
            }

            fn varying_len_matches(
                columns: &Self::VaryingColumns<'_>,
                row_count: usize,
            ) -> bool {
                $($t::varying_len(&columns.$idx) == row_count &&)+ true
            }

            fn decoded_lens_match(columns: &Self::Columns, row_count: usize) -> bool {
                $(columns.$idx.addresses_rows(row_count) &&)+ true
            }

            fn get_varying<'a>(
                columns: &Self::VaryingColumns<'a>,
                index: usize,
            ) -> Self::Elems<'a> {
                ($($t::get_varying(&columns.$idx, index),)+)
            }

            unsafe fn get_varying_unchecked<'a>(
                columns: &Self::VaryingColumns<'a>,
                index: usize,
            ) -> Self::Elems<'a> {
                // SAFETY: forwarded from this method's contract.
                ($(unsafe { $t::get_varying_unchecked(&columns.$idx, index) },)+)
            }

            fn constants(columns: &Self::Columns) -> Self::ConstElems<'_> {
                ($(columns.$idx.constant(),)+)
            }
        }
    };
}

element_tuple!(1; A:0);
element_tuple!(2; A:0, B:1);
element_tuple!(3; A:0, B:1, C:2);
element_tuple!(4; A:0, B:1, C:2, D:3);
element_tuple!(5; A:0, B:1, C:2, D:3, E:4);
element_tuple!(6; A:0, B:1, C:2, D:3, E:4, F:5);
element_tuple!(7; A:0, B:1, C:2, D:3, E:4, F:5, G:6);
element_tuple!(8; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
element_tuple!(9; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
element_tuple!(10; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
element_tuple!(11; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
element_tuple!(12; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);

impl<T: NativePType> IndexedElementTuple for (T,) {
    type Source<'a> = UnaryTupleSource<'a, T>;

    fn indexed_source<'a>(columns: &Self::VaryingColumns<'a>) -> Self::Source<'a> {
        UnaryTupleSource(columns.0)
    }
}

impl<Left: NativePType, Right: NativePType> IndexedElementTuple for (Left, Right) {
    type Source<'a> = LaneZip<&'a [Left], &'a [Right]>;

    fn indexed_source<'a>(columns: &Self::VaryingColumns<'a>) -> Self::Source<'a> {
        LaneZip::new(columns.0, columns.1)
    }
}

#[cfg(test)]
mod tests {
    use vortex_compute::lane_kernels::IndexedSource;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_mask::Mask;

    use super::UnaryTupleSource;
    use super::batch_constant;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::MaskedArray;
    use crate::dtype::Nullability;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::validity::Validity;

    #[test]
    fn test_unary_tuple_source_reads_one_tuple_per_row() {
        let source = UnaryTupleSource(&[10, 20, 30]);
        assert_eq!(source.len(), 3);

        // SAFETY: index one is within the three-element source.
        assert_eq!(unsafe { source.get_unchecked(1) }, (20,));
    }

    #[test]
    fn test_batch_constant_unwraps_filtered_masked_constant() -> VortexResult<()> {
        let child = ConstantArray::new(7_i64, 3).into_array();
        let masked =
            MaskedArray::try_new(child, Validity::from_iter([true, false, true]))?.into_array();
        let filtered = masked.filter(Mask::from_iter([true, true, false]))?;

        let Some(constant) = batch_constant(&filtered) else {
            vortex_bail!("filtered masked constant must remain batch-constant");
        };

        assert!(constant.as_constant().is_some());
        Ok(())
    }

    #[test]
    fn test_batch_constant_preserves_filtered_extension() -> VortexResult<()> {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let extension =
            ExtensionArray::new(ext_dtype, ConstantArray::new(7_i64, 3).into_array()).into_array();
        let filtered = extension.filter(Mask::from_iter([true, false, true]))?;

        let Some(constant) = batch_constant(&filtered) else {
            vortex_bail!("filtered extension storage must remain batch-constant");
        };

        assert_eq!(constant.dtype(), extension.dtype());
        Ok(())
    }
}
