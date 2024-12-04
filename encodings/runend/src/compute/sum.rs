use std::cmp::{max, min};
use std::fmt::{Debug, Display};

use num_traits::Num;
use vortex_array::compute::SumFn;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, IntoCanonical as _};
use vortex_dtype::half::f16;
use vortex_dtype::{match_each_native_ptype, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

fn sum<I: F64It + Num + Copy + Display, V: F64It + Num + Copy>(
    run_ends: &[I],
    values: &[V],
    window_ends: &[u64],
) -> VortexResult<ArrayData>
where
    u64: From<I>,
    for<'a> &'a [I]: Debug,
    for<'a> &'a [V]: Debug,
{
    let mut output = Vec::<f64>::with_capacity(window_ends.len());
    let mut window_start = window_ends[0];
    let mut run_start = 0;
    let mut run_index = 0;
    for window_end in window_ends.iter().skip(1) {
        let window_end = *window_end;

        assert!(window_start < window_end);

        while u64::from(run_ends[run_index]) < window_start {
            run_start = u64::from(run_ends[run_index]);
            run_index += 1;
        }
        let mut sum = 0_f64;

        while run_start < window_end {
            let run_end = u64::from(run_ends[run_index]);

            let sliced_run_start = max(window_start, run_start);
            let sliced_run_end = min(window_end, run_end);
            let slice_len = sliced_run_end - sliced_run_start;

            sum += slice_len.f64it() * values[run_index].f64it();

            run_start = run_end;
            run_index += 1;
        }

        output.push(sum);
        window_start = window_end;
    }
    Ok(ArrayData::from(output))
}

impl SumFn<RunEndArray> for RunEndEncoding {
    fn sum(&self, array: &RunEndArray, window_ends: &[u64]) -> VortexResult<ArrayData> {
        let ends = array.ends().into_canonical()?.into_primitive()?;
        let values = array.values().into_canonical()?.into_primitive()?;
        match_each_unsigned_integer_ptype!(ends.ptype(), |$P| {
                let ends = ends.maybe_null_slice::<$P>();
                match_each_native_ptype!(values.ptype(), |$V| {
                    let values = values.maybe_null_slice::<$V>();
                    sum(ends, values, window_ends)
                })
        })
    }
}

trait F64It {
    fn f64it(self) -> f64;
}

impl F64It for u8 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for u16 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for u32 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for u64 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for i8 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for i16 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for i32 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for i64 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for f16 {
    fn f64it(self) -> f64 {
        self.to_f64()
    }
}
impl F64It for f32 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
impl F64It for f64 {
    fn f64it(self) -> f64 {
        self as f64
    }
}
