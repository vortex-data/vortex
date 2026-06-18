# Explore and Query

## Browse

Open the converted taxi data in the interactive TUI:

```bash
vx browse yellow_tripdata_2024-01.vortex
```

The browser has three tabs (cycle with `Tab`):

**Layout** — Browse the hierarchical layout structure of the file. Use `hjkl` or arrow keys to
navigate, `/` to fuzzy-search for a specific layout node.

**Segments** — Visual grid showing how segments are laid out in the file, with byte offsets and
row ranges.

**Query** — A full SQL console powered by DataFusion. Write queries against a table called `data`:

```sql
SELECT
    PULocationID,
    COUNT(*) as num_trips,
    ROUND(AVG(trip_distance), 2) as avg_distance,
    ROUND(AVG(total_amount), 2) as avg_total
FROM data
GROUP BY PULocationID
ORDER BY num_trips DESC
LIMIT 10
```

Use `[` and `]` to paginate results, `s` to sort by the selected column.

For a browser-based view of a single Vortex file, use
[Vortex Explorer](https://explore.vortex.dev/). It lets you visually inspect the file contents,
including raw bytes in a hex dump-like view.

## Inspect

View the file structure without the full TUI:

```bash
# Show the encoding tree
vx tree layout yellow_tripdata_2024-01.vortex

# Show detailed segment layout
vx segments yellow_tripdata_2024-01.vortex

# Inspect file metadata (EOF marker, postscript, footer)
vx inspect yellow_tripdata_2024-01.vortex
```

All inspection commands support `--json` for machine-readable output.

## Standalone SQL

For non-interactive use, the `vx query` command runs a SQL query and outputs JSON:

```bash
vx query yellow_tripdata_2024-01.vortex \
    --sql "SELECT COUNT(*) as num_trips, ROUND(SUM(total_amount), 2) as total_revenue FROM data"
```

This is useful for scripting and piping results to other tools.
