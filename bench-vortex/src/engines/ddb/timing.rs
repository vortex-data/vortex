use std::time::Duration;

use anyhow::bail;
use regex::{Captures, Regex};

pub struct QueryDuration {
    pub real: Duration,
    pub user: Duration,
    pub sys: Duration,
}

pub fn parse_query_output(output: &str) -> anyhow::Result<QueryDuration> {
    let unpack_time = |c: &Captures, idx: usize| {
        let Some(capture) = c.get(1) else {
            bail!("no capture at idx {idx}");
        };
        let value: f64 = capture.as_str().parse()?;
        Ok(value)
    };

    let pattern = Regex::new("^Run\\sTime\\s\\(s\\):\\sreal\\s(\\d+\\.\\d+)\\suser\\s(\\d+\\.\\d+)\\ssys\\s(\\d+\\.\\d+)").expect("err building regex");

    let Some(captures) = pattern.captures(output) else {
        bail!("pattern didn't match output ({})", output)
    };

    let real = unpack_time(&captures, 1)?;
    let user = unpack_time(&captures, 2)?;
    let sys = unpack_time(&captures, 3)?;

    Ok(QueryDuration {
        real: Duration::from_secs_f64(real),
        user: Duration::from_secs_f64(user),
        sys: Duration::from_secs_f64(sys),
    })
}
