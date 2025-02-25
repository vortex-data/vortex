use vortex_datetime_dtype::TimeUnit;
use vortex_error::{VortexResult, vortex_bail};

pub const SECONDS_PER_DAY: i64 = 86_400; // 24 * 60 * 60

/// Parts of a Unix timestamp (time since 1970-01-01 00:00:00 UTC).
///
/// Broken down into:
///  * Days since epoch
///  * Seconds within day (0-86399)
///  * Subseconds (range depends on `TimeUnit`)
pub struct TimestampParts {
    pub days: i64,
    pub seconds: i64,
    pub subseconds: i64,
}

/// Splits a Unix timestamp into its component parts.
///
/// # Arguments
///
/// * `timestamp` - Number of time units since the Unix epoch
/// * `time_unit` - Precision of the timestamp (ns, μs, ms, or s)
///
/// # Errors
///
/// Returns an error if `time_unit` is days, which cannot be split.
pub fn split(timestamp: i64, time_unit: TimeUnit) -> VortexResult<TimestampParts> {
    let divisor = match time_unit {
        TimeUnit::Ns => 1_000_000_000,
        TimeUnit::Us => 1_000_000,
        TimeUnit::Ms => 1_000,
        TimeUnit::S => 1,
        TimeUnit::D => vortex_bail!("Cannot handle day-level data"),
    };

    let ticks_per_day = SECONDS_PER_DAY * divisor;
    Ok(TimestampParts {
        days: timestamp / ticks_per_day,
        seconds: (timestamp % ticks_per_day) / divisor,
        subseconds: (timestamp % ticks_per_day) % divisor,
    })
}

/// Combines timestamp parts back into a Unix timestamp.
///
/// # Arguments
///
/// * `parts` - Component parts of the timestamp (days, seconds, subseconds)
/// * `time_unit` - Precision of the timestamp (ns, μs, ms, or s)
///
/// # Errors
///
/// Returns an error if `time_unit` is days, which cannot be combined.
pub fn combine(ts_parts: TimestampParts, time_unit: TimeUnit) -> VortexResult<i64> {
    let divisor = match time_unit {
        TimeUnit::Ns => 1_000_000_000,
        TimeUnit::Us => 1_000_000,
        TimeUnit::Ms => 1_000,
        TimeUnit::S => 1,
        TimeUnit::D => vortex_bail!("Cannot handle day-level data"),
    };

    Ok(
        ts_parts.days * SECONDS_PER_DAY * divisor
            + ts_parts.seconds * divisor
            + ts_parts.subseconds,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_seconds() {
        // 1970-01-02 01:02:03 UTC
        let ts = SECONDS_PER_DAY + 3723; // 1 day + (1*3600 + 2*60 + 3) seconds
        let parts = split(ts, TimeUnit::S).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 0);
    }

    #[test]
    fn test_split_milliseconds() {
        // 1970-01-02 01:02:03.456 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1000 + 456;
        let parts = split(ts, TimeUnit::Ms).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456);
    }

    #[test]
    fn test_split_microseconds() {
        // 1970-01-02 01:02:03.456789 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1_000_000 + 456789;
        let parts = split(ts, TimeUnit::Us).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456789);
    }

    #[test]
    fn test_split_nanoseconds() {
        // 1970-01-02 01:02:03.456789123 UTC
        let ts = (SECONDS_PER_DAY + 3723) * 1_000_000_000 + 456789123;
        let parts = split(ts, TimeUnit::Ns).unwrap();
        assert_eq!(parts.days, 1);
        assert_eq!(parts.seconds, 3723);
        assert_eq!(parts.subseconds, 456789123);
    }

    #[test]
    fn test_split_epoch() {
        let ts = 0;
        for unit in [TimeUnit::S, TimeUnit::Ms, TimeUnit::Us, TimeUnit::Ns] {
            let parts = split(ts, unit).unwrap();
            assert_eq!(parts.days, 0);
            assert_eq!(parts.seconds, 0);
            assert_eq!(parts.subseconds, 0);
        }
    }

    #[test]
    fn test_split_days_error() {
        assert!(split(1, TimeUnit::D).is_err());
    }

    #[test]
    fn test_combine_seconds() {
        // 1970-01-02 01:02:03 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723, // 1*3600 + 2*60 + 3
            subseconds: 0,
        };
        let ts = combine(parts, TimeUnit::S).unwrap();
        assert_eq!(ts, SECONDS_PER_DAY + 3723);
    }

    #[test]
    fn test_combine_milliseconds() {
        // 1970-01-02 01:02:03.456 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456,
        };
        let ts = combine(parts, TimeUnit::Ms).unwrap();
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1000 + 456);
    }

    #[test]
    fn test_combine_microseconds() {
        // 1970-01-02 01:02:03.456789 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456789,
        };
        let ts = combine(parts, TimeUnit::Us).unwrap();
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1_000_000 + 456789);
    }

    #[test]
    fn test_combine_nanoseconds() {
        // 1970-01-02 01:02:03.456789123 UTC
        let parts = TimestampParts {
            days: 1,
            seconds: 3723,
            subseconds: 456789123,
        };
        let ts = combine(parts, TimeUnit::Ns).unwrap();
        assert_eq!(ts, (SECONDS_PER_DAY + 3723) * 1_000_000_000 + 456789123);
    }

    #[test]
    fn test_combine_days_error() {
        let parts = TimestampParts {
            days: 1,
            seconds: 0,
            subseconds: 0,
        };
        assert!(combine(parts, TimeUnit::D).is_err());
    }
}
