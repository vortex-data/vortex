// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`]s, and the per-argument decode behind them.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Extension;
use crate::arrays::Masked;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::InputElement;

mod private {
    pub trait Sealed {}
}

/// One decoded input column of an [`ElementTuple`].
///
/// A constant operand holds the same value in every row, so it is decoded once as a single row and
/// read at index 0 forever. That is what stops a constant argument costing one decode per row, which
/// matters whenever the decode is more than a buffer read: parsing a geometry from WKB, or
/// canonicalizing an extension row.
pub struct ArgColumn<T: InputElement>(ArgColumnKind<T>);

enum ArgColumnKind<T: InputElement> {
    Varying(T::Column),
    Constant(T::Column),
}

impl<T: InputElement> ArgColumn<T> {
    /// Decode one input column, collapsing a constant operand to its single distinct row.
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

    /// Like [`decode`](Self::decode), but a varying column decodes null-tolerantly through
    /// [`InputElement::decode_null_tolerant`]. `Ok(None)` means the element cannot, and the
    /// caller falls back to the filter strategy.
    ///
    /// A constant operand still takes the ordinary decode: the lifting short-circuits null
    /// constants before any strategy runs, so a constant reaching here is non-null.
    fn decode_null_tolerant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
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

    /// Read the element at `index`, which for a constant operand is always its single row.
    fn get(&self, index: usize) -> T::Elem<'_> {
        match &self.0 {
            ArgColumnKind::Varying(column) => T::get(column, index),
            ArgColumnKind::Constant(column) => T::get(column, 0),
        }
    }

    /// The decoded full column, or `None` when this argument was collapsed to one constant row.
    fn varying(&self) -> Option<&T::Column> {
        match &self.0 {
            ArgColumnKind::Varying(column) => Some(column),
            ArgColumnKind::Constant(_) => None,
        }
    }

    /// Whether this argument addresses exactly `row_count` rows.
    ///
    /// A constant operand was collapsed to its one distinct row and is read at index 0 forever, so
    /// it addresses any row count and is exempt.
    fn addresses_rows(&self, row_count: usize) -> bool {
        match &self.0 {
            ArgColumnKind::Varying(column) => T::varying_len(&T::varying(column)) == row_count,
            ArgColumnKind::Constant(_) => true,
        }
    }

    /// The single decoded element of a constant operand, or `None` for a real column.
    ///
    /// `Some` exactly when [`decode`](Self::decode) collapsed the operand to its one distinct row,
    /// in which case the value returned is the element every row of the batch reads.
    fn constant(&self) -> Option<T::Elem<'_>> {
        match &self.0 {
            ArgColumnKind::Varying(_) => None,
            ArgColumnKind::Constant(column) => Some(T::get(column, 0)),
        }
    }
}

/// The array whose every row holds one distinct value, when `array` is constant for the batch.
///
/// Beyond the constant encoding itself this sees one level through two wrappers that spell "the
/// same value in every row" without being the constant encoding:
///
/// - [`Masked`], how the compressor spells an all-same-with-nulls chunk: the child carries the
///   value, the wrapper carries only validity. Reading the child's value for a null row is sound
///   here because the lifting owns validity entirely; the row loop's output behind a null
///   row is masked away (dense) or never computed (filter), so which value the loop read there
///   cannot be observed. An all-null constant never reaches decode at all, since the lifting
///   short-circuits it to an all-null result first.
/// - [`Extension`] over constant storage, the shape an extension-typed builder produces before
///   `ExtensionConstantRule` normalizes it to a top-level constant. Every row wraps the same
///   storage value, so the whole array (sliced to one row, keeping its extension dtype) is the
///   constant.
pub(in crate::scalar_fn::row) fn batch_constant(array: &ArrayRef) -> Option<ArrayRef> {
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

/// Tuples of [`InputElement`]s forming the typed argument list a [`RowFn`](crate::scalar_fn::RowFn)
/// visits with. Implemented for `()` and tuples of one through twelve elements. This trait is
/// framework-only; add a new decode primitive by implementing [`InputElement`], then use it inside
/// one of those tuples.
///
/// The arities past the widest function in tree are deliberate. This trait is **sealed**, so a
/// downstream crate cannot add the one it needs, and an unused arity costs only its own macro
/// expansion: no monomorphization happens until something instantiates it.
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
    /// [`visit_prepared_into`](crate::scalar_fn::RowVisitor::visit_prepared_into) hands to its prepare
    /// closure, so a kernel can hoist work that depends only on a constant argument out of the
    /// row loop.
    type ConstElems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether _any_ argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// The additive cost of per-row decode work avoided by filtering the arguments first.
    const FILTERED_DECODE_COST: usize;

    /// Validate the input dtypes, including that `dtypes` has exactly `ARITY` entries.
    ///
    /// The expression layer checks the count against [`Arity`](crate::scalar_fn::Arity) before it
    /// builds a call, but this is also the entry point of the public
    /// [`return_dtype`](crate::scalar_fn::ScalarFnVTable::return_dtype), so the count is enforced
    /// here rather than assumed.
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once. Called once per batch.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Decode every input column once, tolerating null rows, or `Ok(None)` when some argument
    /// cannot. Called once per batch by the branch-and-skip null strategy.
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

    /// Read the batch-constant elements out of the decoded columns. Called once per batch.
    fn constants(columns: &Self::Columns) -> Self::ConstElems<'_>;
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
    const FILTERED_DECODE_COST: usize = 0;

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
            const FILTERED_DECODE_COST: usize = $($t::FILTERED_DECODE_COST +)+ 0;

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
