# Vortex Dataset API

The Vortex Dataset API exposes an interoperable interface for scanning data. It is designed to solve the `NxM` problem
of integration query engines with data formats.

It is defined as a C API with language bindings for Rust, Java, and Python. This allows both data sources and query
engines to be implemented in any language that can interoperate with C.

The API passes data in the form of Vortex ArrayStreams, which are zero-copy, columnar, compressed, and even support
passing device buffers. This allows query engines to efficiently scan data into their internal execution formats with
minimal overhead and (very) late materialization.

Known implementations of the Vortex Dataset API are:

* `vortex-iceberg` - Expose Iceberg tables as a Vortex Dataset
* `vortex-python` - Expose PyArrow Datasets as a Vortex Dataset
* `vortex-layout` - Expose a Vortex Layout as a Vortex Dataset

Known consumers of the Vortex Dataset API are:

* `vortex-datafusion` - Scan Vortex Datasets in DataFusion
* `vortex-duckdb` - Scan Vortex Datasets in DuckDB
* `vortex-spark` - Scan Vortex Datasets in Spark
* `vortex-trino` - Scan Vortex Datasets in Trino
* `vortex-polars` - Scan Vortex Datasets in Polars
* `vortex-python` - Wrap Vortex Datasets as PyArrow Datasets

╔═══════════════════════════════════════════════════════════════════════════════╗
║  WITH VORTEX DATASET API (N+M integrations = 9 connections)                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   DATA SOURCES              VORTEX API               QUERY ENGINES            ║
║                                                                               ║
║   ┌──────────────┐      ╔═══════════════╗       ┌────────────────┐            ║
║   │   Iceberg    │─────▶║               ║──────▶│   DataFusion   │            ║
║   │ (vortex-     │      ║   Vortex      ║       │(vortex-        │            ║
║   │   iceberg)   │      ║   Dataset     ║       │  datafusion)   │            ║
║   └──────────────┘      ║   API         ║       └────────────────┘            ║
║                         ║               ║                                     ║
║   ┌──────────────┐      ║ ┌───────────┐ ║       ┌────────────────┐            ║
║   │   PyArrow    │─────▶║ │ • C ABI   │ ║──────▶│     DuckDB     │            ║
║   │   Datasets   │      ║ │ • Zero-   │ ║       │ (vortex-duckdb)│            ║
║   │ (vortex-     │      ║ │   copy    │ ║       └────────────────┘            ║
║   │   python)    │      ║ │ • Columnar│ ║                                     ║
║   └──────────────┘      ║ │ • Compress│ ║       ┌────────────────┐            ║
║                         ║ │ • Device  │ ║──────▶│     Spark      │            ║
║   ┌──────────────┐      ║ │   buffers │ ║       │ (vortex-spark) │            ║
║   │    Vortex    │─────▶║ └───────────┘ ║       └────────────────┘            ║
║   │    Layout    │      ║               ║                                     ║
║   │  (vortex-    │      ║  Language     ║       ┌────────────────┐            ║
║   │    layout)   │      ║  Bindings:    ║──────▶│     Trino      │            ║
║   └──────────────┘      ║  Rust│Java│Py ║       │ (vortex-trino) │            ║
║                         ║               ║       └────────────────┘            ║
║                         ║               ║                                     ║
║                         ║               ║       ┌────────────────┐            ║
║                         ║               ║──────▶│     Polars     │            ║
║                         ║               ║       │(vortex-polars) │            ║
║                         ║               ║       └────────────────┘            ║
║                         ║               ║                                     ║
║                         ║               ║       ┌────────────────┐            ║
║                         ║               ║──────▶│     Python     │            ║
║                         ╚═══════════════╝       │(vortex-python) │            ║
║                                                 └────────────────┘            ║
║                                                                               ║
║ ✅ Add a new source? All engines get it. Add a new engine? All sources work!  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
