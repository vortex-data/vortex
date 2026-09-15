// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run with `cargo run -p vortex-pco --example probe`.

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexResult;
use vortex_fastlanes::RLEData;
use vortex_pco::Pco;
use vortex_runend::RunEnd;

fn main() -> VortexResult<()> {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    vortex_runend::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values =
        PrimitiveArray::from_option_iter((0..4096u32).map(|i| (i % 11 != 0).then_some(i / 16)));
    let rle = RLEData::encode(values.as_view(), &mut ctx)?.into_array();
    let pco = Pco::from_primitive(values.as_view(), 3, 1024, &mut ctx)?.into_array();
    let ends = PrimitiveArray::from_iter((1..=4096u32).map(|run| run * 4));
    let ends = Pco::from_primitive(ends.as_view(), 3, 1024, &mut ctx)?.into_array();
    let runend_pco = RunEnd::try_new(ends, pco.clone(), &mut ctx)?.into_array();

    for (name, array) in [("RLE", rle), ("PCO", pco), ("RunEnd(PCO, PCO)", runend_pco)] {
        // A one-off probe borrows the array and retains nothing.
        let scalar = array.probe().execute_scalar(17, &mut ctx)?;
        println!("{name} single lookup: {scalar}");

        // A repeated probe keeps child probes; each PCO child keeps its own decoded page.
        let mut probe = array.repeated_probe();
        for index in [17, 33, 22, 1025, 17] {
            println!(
                "{name}[{index}] = {}",
                probe.execute_scalar(index, &mut ctx)?
            );
        }
    }
    Ok(())
}
