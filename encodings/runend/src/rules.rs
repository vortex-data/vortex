// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::AnyScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnVTable;
use vortex_array::dtype::DType;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use vortex_error::VortexResult;

use crate::RunEnd;

pub(super) const RULES: ParentRuleSet<RunEnd> = ParentRuleSet::new(&[
    // CastReduceAdaptor must come before RunEndScalarFnRule so that cast operations are executed
    // eagerly (surfacing out-of-range errors immediately) rather than being pushed lazily into
    // the values array by the generic scalar function push-down rule.
    ParentRuleSet::lift(&CastReduceAdaptor(RunEnd)),
    ParentRuleSet::lift(&RunEndScalarFnRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(RunEnd)),
]);

/// A rule to push down scalar functions through run-end encoding into the values array.
///
/// This only works if all other children of the scalar function array are constants.
#[derive(Debug)]
pub(crate) struct RunEndScalarFnRule;

impl ArrayParentReduceRule<RunEnd> for RunEndScalarFnRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        run_end: ArrayView<'_, RunEnd>,
        parent: ArrayView<'_, ScalarFnVTable>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for (idx, child) in parent.iter_children().enumerate() {
            if idx == child_idx {
                // Skip ourselves
                continue;
            }

            if !child.is::<Constant>() {
                // We can only push down if all other children are constants
                return Ok(None);
            }
        }

        if !matches!(
            parent.dtype(),
            DType::Bool(_) | DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
        ) {
            return Ok(None);
        }

        let values_len = run_end.values().len();
        let mut new_children: Vec<ArrayRef> = parent.children().to_vec();
        for (idx, child) in new_children.iter_mut().enumerate() {
            if idx == child_idx {
                // Replace ourselves with run end values
                *child = run_end.values().clone();
                continue;
            }

            // Replace other children with their constant scalar value with length adjusted
            // to the length of the run end values.
            let constant = child.as_::<Constant>();
            *child = ConstantArray::new(constant.scalar().clone(), values_len).into_array();
        }

        let new_values =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, values_len)?
                .into_array();

        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    run_end.ends().clone(),
                    new_values,
                    run_end.offset(),
                    run_end.len(),
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use super::*;
    use crate::RunEnd;

    fn bool_mask_fixture() -> ArrayRef {
        RunEnd::new(
            buffer![256u32, 512, 768, 1024].into_array(),
            BoolArray::from_iter([true, false, true, false]).into_array(),
        )
        .into_array()
    }

    #[test]
    fn pushes_down_utf8_zip_to_runend() {
        let mask = bool_mask_fixture();
        let if_true = ConstantArray::new(
            Scalar::utf8("runend-true-branch", Nullability::NonNullable),
            mask.len(),
        )
        .into_array();
        let if_false = ConstantArray::new(
            Scalar::utf8("runend-false-branch", Nullability::NonNullable),
            mask.len(),
        )
        .into_array();

        let optimized = mask.zip(if_true, if_false).unwrap().optimize().unwrap();

        assert!(optimized.is::<RunEnd>());
        assert_eq!(optimized.dtype(), &DType::Utf8(Nullability::NonNullable));

        let actual = optimized
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .0
            .into_array();
        let expected = vortex_array::arrays::VarBinViewArray::from_iter_str((0..1024).map(|idx| {
            if idx < 256 || (512..768).contains(&idx) {
                "runend-true-branch"
            } else {
                "runend-false-branch"
            }
        }))
        .into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn pushes_down_binary_zip_to_runend() {
        let mask = bool_mask_fixture();
        let if_true = vec![0xAA; 8];
        let if_false = vec![0x55; 12];
        let optimized = mask
            .zip(
                ConstantArray::new(
                    Scalar::binary(if_true.clone(), Nullability::NonNullable),
                    mask.len(),
                )
                .into_array(),
                ConstantArray::new(
                    Scalar::binary(if_false.clone(), Nullability::NonNullable),
                    mask.len(),
                )
                .into_array(),
            )
            .unwrap()
            .optimize()
            .unwrap();

        assert!(optimized.is::<RunEnd>());
        assert_eq!(optimized.dtype(), &DType::Binary(Nullability::NonNullable));

        let actual = optimized
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .0
            .into_array();
        let expected = vortex_array::arrays::VarBinViewArray::from_iter_bin((0..1024).map(|idx| {
            if idx < 256 || (512..768).contains(&idx) {
                if_true.clone()
            } else {
                if_false.clone()
            }
        }))
        .into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn pushes_down_sliced_nullable_utf8_zip_to_runend() {
        let mask = bool_mask_fixture()
            .slice(128..896)
            .unwrap()
            .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let optimized = mask
            .zip(
                ConstantArray::new(
                    Scalar::utf8("slice-true-branch", Nullability::Nullable),
                    mask.len(),
                )
                .into_array(),
                ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), mask.len())
                    .into_array(),
            )
            .unwrap()
            .optimize()
            .unwrap();

        assert!(optimized.is::<RunEnd>());
        assert_eq!(optimized.dtype(), &DType::Utf8(Nullability::Nullable));
        assert_eq!(optimized.as_::<RunEnd>().offset(), 128);

        let actual = optimized
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .0
            .into_array();
        let expected = vortex_array::arrays::VarBinViewArray::from_iter(
            (128..896)
                .map(|idx| (idx < 256 || (512..768).contains(&idx)).then_some("slice-true-branch")),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn keeps_struct_zip_on_fallback_path() {
        let mask = bool_mask_fixture();
        let struct_dtype = DType::Struct(
            StructFields::new(
                FieldNames::from(["value"]),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let if_true = ConstantArray::new(
            Scalar::struct_(struct_dtype.clone(), vec![Scalar::from(1i32)]),
            mask.len(),
        )
        .into_array();
        let if_false = ConstantArray::new(
            Scalar::struct_(struct_dtype, vec![Scalar::from(2i32)]),
            mask.len(),
        )
        .into_array();

        let optimized = mask.zip(if_true, if_false).unwrap().optimize().unwrap();

        assert!(!optimized.is::<RunEnd>());
    }
}
