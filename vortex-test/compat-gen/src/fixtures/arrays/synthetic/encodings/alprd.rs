// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::RDEncoder;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct AlprdFixture;

impl FlatLayoutFixture for AlprdFixture {
    fn name(&self) -> &str {
        "alprd.vortex"
    }

    fn description(&self) -> &str {
        "Real-valued doubles with small deltas for ALPRD encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ALPRD::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let sensor: Vec<f64> = (0..N)
            .map(|i| {
                let noise = ((i * 7 + 13) % 100) as f64 / 1000.0;
                98.6 + noise
            })
            .collect();

        let drift: Vec<f64> = (0..N)
            .map(|i| 1000.0 + (i as f64) * 0.001 + ((i * 3) % 7) as f64 * 0.0001)
            .collect();

        let sensor_nullable_vals: Vec<f64> = (0..N)
            .map(|i| {
                let noise = ((i * 11 + 3) % 100) as f64 / 1000.0;
                37.0 + noise
            })
            .collect();
        let sensor_nullable = PrimitiveArray::from_option_iter((0..N).map(|i| {
            if i % 13 == 0 {
                None
            } else {
                let noise = ((i * 11 + 3) % 100) as f64 / 1000.0;
                Some(37.0 + noise)
            }
        }));

        let sensor_prim = PrimitiveArray::new(Buffer::from(sensor), Validity::NonNullable);
        let drift_prim = PrimitiveArray::new(Buffer::from(drift), Validity::NonNullable);

        let sensor_enc = RDEncoder::new::<f64>(sensor_prim.as_slice::<f64>());
        let drift_enc = RDEncoder::new::<f64>(drift_prim.as_slice::<f64>());
        let nullable_enc = RDEncoder::new::<f64>(&sensor_nullable_vals);

        let arr = StructArray::try_new(
            FieldNames::from(["sensor", "drift", "sensor_nullable"]),
            vec![
                sensor_enc.encode(&sensor_prim).into_array(),
                drift_enc.encode(&drift_prim).into_array(),
                nullable_enc.encode(&sensor_nullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
