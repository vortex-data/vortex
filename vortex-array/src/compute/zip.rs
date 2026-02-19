// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;

/// Performs element-wise conditional selection between two arrays based on a mask.
///
/// Returns a new array where `result[i] = if_true[i]` when `mask[i]` is true,
/// otherwise `result[i] = if_false[i]`.
///
/// Null values in the mask are treated as false (selecting `if_false`). This follows
/// SQL semantics (DuckDB, Trino) where a null condition falls through to the ELSE branch,
/// rather than Arrow's `if_else` which propagates null conditions to the output.
pub fn zip(if_true: &dyn Array, if_false: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    if_true
        .to_array()
        .zip(if_false.to_array(), mask.clone().into_array())
}

pub(crate) fn zip_return_dtype(if_true: &dyn Array, if_false: &dyn Array) -> DType {
    if_true
        .dtype()
        .union_nullability(if_false.dtype().nullability())
}

pub(crate) fn zip_impl(
    if_true: &dyn Array,
    if_false: &dyn Array,
    mask: &Mask,
) -> VortexResult<ArrayRef> {
    assert_eq!(
        if_true.len(),
        if_false.len(),
        "zip requires arrays to have the same size"
    );

    let return_type = zip_return_dtype(if_true, if_false);
    zip_impl_with_builder(
        if_true,
        if_false,
        mask,
        builder_with_capacity(&return_type, if_true.len()),
    )
}

fn zip_impl_with_builder(
    if_true: &dyn Array,
    if_false: &dyn Array,
    mask: &Mask,
    mut builder: Box<dyn ArrayBuilder>,
) -> VortexResult<ArrayRef> {
    match mask.slices() {
        AllOr::All => Ok(if_true.to_array()),
        AllOr::None => Ok(if_false.to_array()),
        AllOr::Some(slices) => {
            for (start, end) in slices {
                builder.extend_from_array(&if_false.slice(builder.len()..*start)?);
                builder.extend_from_array(&if_true.slice(*start..*end)?);
            }
            if builder.len() < if_false.len() {
                builder.extend_from_array(&if_false.slice(builder.len()..if_false.len())?);
            }
            Ok(builder.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_select::zip::zip as arrow_zip;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewVTable;
    use crate::arrow::IntoArrowArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::BufferGrowthStrategy;
    use crate::builders::VarBinViewBuilder;
    use crate::compute::zip;
    use crate::scalar::Scalar;

    #[test]
    fn test_zip_basic() {
        let mask = Mask::from_iter([true, false, false, true, false]);
        let if_true = buffer![10, 20, 30, 40, 50].into_array();
        let if_false = buffer![1, 2, 3, 4, 5].into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();
        let expected = buffer![10, 2, 3, 40, 5].into_array();

        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_zip_all_true() {
        let mask = Mask::new_true(4);
        let if_true = buffer![10, 20, 30, 40].into_array();
        let if_false =
            PrimitiveArray::from_option_iter([Some(1), Some(2), Some(3), None]).into_array();

        let result = zip(&if_true, &if_false, &mask).unwrap();
        let expected =
            PrimitiveArray::from_option_iter([Some(10), Some(20), Some(30), Some(40)]).into_array();

        assert_arrays_eq!(result, expected);

        // result must be nullable even if_true was not
        assert_eq!(result.dtype(), if_false.dtype())
    }

    #[test]
    #[should_panic]
    fn test_invalid_lengths() {
        let mask = Mask::new_false(4);
        let if_true = buffer![10, 20, 30].into_array();
        let if_false = buffer![1, 2, 3, 4].into_array();

        zip(&if_true, &if_false, &mask).unwrap();
    }

    #[test]
    fn test_fragmentation() {
        let len = 100;

        let const1 = ConstantArray::new(
            Scalar::utf8("hello_this_is_a_longer_string", Nullability::Nullable),
            len,
        )
        .to_array();

        let const2 = ConstantArray::new(
            Scalar::utf8("world_this_is_another_string", Nullability::Nullable),
            len,
        )
        .to_array();

        // Create a mask that alternates frequently to cause fragmentation
        // Pattern: take from const1 at even indices, const2 at odd indices
        let indices: Vec<usize> = (0..len).step_by(2).collect();
        let mask = Mask::from_indices(len, indices);

        let result = zip(&const1, &const2, &mask).unwrap();

        insta::assert_snapshot!(result.display_tree(), @r"
        root: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
          metadata: EmptyMetadata
          buffer: buffer_0 host 29 B (align=1) (1.75%)
          buffer: buffer_1 host 28 B (align=1) (1.69%)
          buffer: views host 1.60 kB (align=16) (96.56%)
        ");

        // test wrapped in a struct
        let wrapped1 = StructArray::try_from_iter([("nested", const1)])
            .unwrap()
            .to_array();
        let wrapped2 = StructArray::try_from_iter([("nested", const2)])
            .unwrap()
            .to_array();

        let wrapped_result = zip(&wrapped1, &wrapped2, &mask).unwrap();
        insta::assert_snapshot!(wrapped_result.display_tree(), @r"
        root: vortex.struct({nested=utf8?}, len=100) nbytes=1.66 kB (100.00%)
          metadata: EmptyMetadata
          nested: vortex.varbinview(utf8?, len=100) nbytes=1.66 kB (100.00%) [all_valid]
            metadata: EmptyMetadata
            buffer: buffer_0 host 29 B (align=1) (1.75%)
            buffer: buffer_1 host 28 B (align=1) (1.69%)
            buffer: views host 1.60 kB (align=16) (96.56%)
        ");
    }

    #[test]
    fn test_varbinview_zip() {
        let if_true = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello");
                builder.append_value("Hello this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        let if_false = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
                0.0,
            );
            for _ in 0..100 {
                builder.append_value("Hello2");
                builder.append_value("Hello2 this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        // [1,2,4,5,7,8,..]
        let mask = Mask::from_indices(200, (0..100).filter(|i| i % 3 != 0).collect());

        let zipped = zip(&if_true, &if_false, &mask).unwrap();
        let zipped = zipped.as_opt::<VarBinViewVTable>().unwrap();
        assert_eq!(zipped.nbuffers(), 2);

        // assert the result is the same as arrow
        let expected = arrow_zip(
            mask.into_array()
                .into_arrow_preferred()
                .unwrap()
                .as_boolean(),
            &if_true.into_arrow_preferred().unwrap(),
            &if_false.into_arrow_preferred().unwrap(),
        )
        .unwrap();

        let actual = zipped.clone().into_array().into_arrow_preferred().unwrap();
        assert_eq!(actual.as_ref(), expected.as_ref());
    }
}
