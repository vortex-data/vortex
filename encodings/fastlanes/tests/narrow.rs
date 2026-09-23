// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(test)]

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Narrow;
use vortex_array::arrays::NarrowArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::narrow::NarrowArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedData;
use vortex_session::registry::ReadContext;

#[test]
fn narrow_bitpacked_comparison_selection_and_roundtrip() -> VortexResult<()> {
    let session = array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter((0u8..32).cycle().take(2300)).into_array();
    let packed = BitPackedData::encode(&primitive, 5, &mut ctx)?.into_array();
    let array = NarrowArray::try_new(packed, PType::U64.into())?;

    let constant = ConstantArray::new(16u64, array.len()).into_array();
    let comparison = Narrow::compare(array.as_view(), &constant, CompareOperator::Lt, &mut ctx)?
        .ok_or_else(|| vortex_err!("expected narrow comparison kernel"))?;
    let comparison_view = comparison.as_::<ScalarFn>();
    assert!(comparison_view.get_child(0).is::<BitPacked>());
    assert_eq!(comparison_view.get_child(1).dtype(), &PType::U8.into());
    assert_arrays_eq!(
        comparison,
        BoolArray::from_iter((0..2300).map(|i| i % 32 < 16)),
        &mut ctx
    );

    let sliced = array.slice(13..2200)?;
    assert!(sliced.as_::<Narrow>().values().is::<BitPacked>());
    let taken = sliced.take(buffer![0u32, 1020, 2186].into_array())?;
    assert!(taken.is::<Narrow>());
    assert_arrays_eq!(taken, buffer![13u64, 9, 23].into_array(), &mut ctx);

    let array = array.into_array();
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &session, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;
    assert!(decoded.as_::<Narrow>().values().is::<BitPacked>());
    assert_eq!(decoded.as_::<Narrow>().values().dtype(), &PType::U8.into());
    assert_arrays_eq!(
        decoded,
        PrimitiveArray::from_iter((0u64..32).cycle().take(2300)),
        &mut ctx
    );
    assert_arrays_eq!(
        decoded.binary(ConstantArray::new(300u64, 2300).into_array(), Operator::Lt)?,
        BoolArray::from_iter([true; 2300]),
        &mut ctx
    );
    Ok(())
}
