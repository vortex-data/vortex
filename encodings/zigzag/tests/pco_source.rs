// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(codspeed))]

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::test_harness::fusion::fusion_trace;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;
    use vortex_session::VortexSession;
    use vortex_zigzag::ZigZag;

    #[rstest]
    #[case::all_valid(false)]
    #[case::nullable(true)]
    fn sliced_pco_pages_fuse(#[case] nullable: bool) -> VortexResult<()> {
        let session = VortexSession::empty();
        vortex_zigzag::initialize(&session);
        vortex_pco::initialize(&session);
        let values = if nullable {
            PrimitiveArray::from_option_iter((0..5127u32).map(|i| (i % 11 != 0).then_some(i / 13)))
        } else {
            PrimitiveArray::from_iter((0..5127u32).map(|i| i / 13))
        };
        let encoded = Pco::from_primitive(
            values.as_view(),
            8,
            521,
            &mut session.create_execution_ctx(),
        )?;
        let array = ZigZag::try_new(encoded.into_array())?.into_array();
        let trace = fusion_trace(array.slice(1027..5011)?, &session)?;
        assert_eq!(
            trace.contains("fuse start array=vortex.pco"),
            nullable,
            "{trace}"
        );
        Ok(())
    }
}
