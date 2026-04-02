// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::like::LikeExecuteAdaptor;

use crate::FSST;

pub(super) const PARENT_KERNELS: ParentKernelSet<FSST> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(FSST)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(FSST)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(FSST)),
    ParentKernelSet::lift(&LikeExecuteAdaptor(FSST)),
]);

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FilterArray;
    use vortex_array::arrays::varbin::builder::VarBinBuilder;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn build_test_fsst_array() -> ArrayRef {
        let mut builder = VarBinBuilder::<i32>::with_capacity(10);
        builder.append_value(b"hello world");
        builder.append_value(b"foo bar baz");
        builder.append_value(b"testing fsst compression");
        builder.append_value(b"another string here");
        builder.append_value(b"the quick brown fox");
        builder.append_value(b"jumps over the lazy dog");
        builder.append_value(b"abcdefghijklmnop");
        builder.append_value(b"qrstuvwxyz");
        builder.append_value(b"0123456789");
        builder.append_value(b"final string");
        let input = builder.finish(DType::Utf8(Nullability::NonNullable));

        let compressor = fsst_train_compressor(&input);
        let len = input.len();
        let dtype = input.dtype().clone();
        fsst_compress(input, len, &dtype, &compressor).into_array()
    }

    #[test]
    fn test_fsst_filter_simple() -> VortexResult<()> {
        let fsst_array = build_test_fsst_array();
        assert!(fsst_array.is::<FSST>());
        assert_eq!(fsst_array.len(), 10);

        // Filter 1/5 elements (every 5th element: indices 0 and 5)
        let mask = Mask::from_iter([
            true, false, false, false, false, true, false, false, false, false,
        ]);

        // Create FilterArray and execute
        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        // Compare with filtering the canonical VarBinView.
        let expected = fsst_array.filter(mask)?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn test_fsst_filter_every_other() -> VortexResult<()> {
        let fsst_array = build_test_fsst_array();

        // Filter every other element
        let mask = Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]);

        let filter_array = FilterArray::new(fsst_array.clone(), mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = fsst_array.filter(mask)?;

        assert_eq!(result.len(), 5);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn issues_6034_test_fsst_filter_with_nulls_and_special_chars() -> VortexResult<()> {
        //
        // Test case with special characters and nulls
        // Values: ["", "", "", "", "", "", "", "", "", "", "", ",", "A<<<<<<<", "", "", "", "", null, null, null, null, null, null]
        // Mask: only the last element is selected (true at index 22)
        let mut builder = VarBinBuilder::<i32>::with_capacity(23);
        // 11 empty strings
        for _ in 0..11 {
            builder.append_value(b"");
        }
        // ","
        builder.append_value(b",");
        // "A<<<<<<<"
        builder.append_value(b"A<<<<<<<");
        // 4 more empty strings
        for _ in 0..4 {
            builder.append_value(b"");
        }
        // 6 nulls
        for _ in 0..6 {
            builder.append_null();
        }
        let input = builder.finish(DType::Utf8(Nullability::Nullable));

        let compressor = fsst_train_compressor(&input);
        let fsst_array: ArrayRef =
            fsst_compress(input.clone(), input.len(), input.dtype(), &compressor).into_array();

        // Filter: only select the last element (index 22)
        let mut mask = vec![false; 22];
        mask.push(true);
        let mask = Mask::from_iter(mask);

        let filter_array = FilterArray::new(fsst_array, mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = input.filter(mask)?;

        assert_eq!(result.len(), 1);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn filter_only_null() -> VortexResult<()> {
        let mut builder = VarBinBuilder::<i32>::with_capacity(3);
        builder.append_null();
        builder.append_value(b"A");
        builder.append_null();

        let input = builder.finish(DType::Utf8(Nullability::Nullable));

        let compressor = fsst_train_compressor(&input);
        let fsst_array: ArrayRef =
            fsst_compress(input.clone(), input.len(), input.dtype(), &compressor).into_array();

        let mask = Mask::from_iter([true, false, true]);

        let filter_array = FilterArray::new(fsst_array, mask.clone()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?;

        let expected = input.filter(mask)?;

        assert_eq!(result.len(), 2);
        assert_arrays_eq!(result.into_array(), expected);
        Ok(())
    }

    #[test]
    fn test_fsst_filter_all_true() -> VortexResult<()> {
        let fsst_array = build_test_fsst_array();
        assert_eq!(fsst_array.len(), 10);

        let mask = Mask::new_true(10);

        let filter_array = fsst_array.filter(mask)?;
        let mut ctx = SESSION.create_execution_ctx();
        let result = filter_array.execute::<Canonical>(&mut ctx)?.into_array();

        assert_arrays_eq!(result, fsst_array);
        Ok(())
    }
}
