// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::execute_sink_valid_rows;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

struct NonSkippingSink;

impl OutputSink for NonSkippingSink {
    type Rows<'a> = ();
    type Row<'a> = ();
    type WriteToken = ();

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(_rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Err(vortex_err!(
            "a non-skipping sink must decline before allocation"
        ))
    }

    fn rows(&mut self) -> Self::Rows<'_> {}

    fn row_count_matches(_rows: &Self::Rows<'_>, _row_count: usize) -> bool {
        true
    }

    fn row<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {}

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Err(vortex_err!("a non-skipping sink must not finish"))
    }
}

#[test]
fn test_non_skipping_sink_declines_before_allocation() -> VortexResult<()> {
    let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let valid = Mask::from_iter([true, false]);
    let mut ctx = array_session().create_execution_ctx();

    let execution = execute_sink_valid_rows::<(i64,), (), NonSkippingSink, ()>(
        &args,
        &DType::from(i64::PTYPE),
        &valid,
        &mut ctx,
        |_| (),
        |_, _, _| (),
    )?;

    assert!(execution.is_none());
    Ok(())
}
