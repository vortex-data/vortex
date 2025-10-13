# Vortex Benchmarks - Modular Architecture

## Overview

The Vortex Benchmarks website has been refactored to use a modular, class-based architecture that provides:
- **Clear separation of concerns** between configuration, logic, and UI
- **Type-specific benchmark classes** for different benchmark categories
- **Centralized configuration** instead of inline HTML configuration
- **Complete backward compatibility** with existing code
- **Consistent behavior** across all query benchmarks
- **Preserved numerical output** - exact same results as before

## Architecture Components

### Core Files

#### 1. **benchmark-types.js**
Defines the class hierarchy for different benchmark types:
- `BaseBenchmark` - Base class with common functionality
- `QueryBenchmark` - For Clickbench, TPC-H, TPC-DS, StatPopGen
- `CompressionBenchmark` - For compression time and size benchmarks
- `RandomAccessBenchmark` - For random access benchmarks

Each class provides:
- Score calculation methods
- Chart configuration options
- Data validation and transformation
- Type-specific behavior

#### 2. **benchmark-config.js**
Centralized configuration for all benchmarks:
- Replaces inline configuration previously in index.html
- Defines benchmark metadata, hidden datasets, renamed datasets, etc.
- Programmatically generates TPC-H and TPC-DS configurations for all scale factors
- Exports `BENCHMARK_CONFIGS` object with all benchmark definitions

#### 3. **benchmark-factory.js**
Factory pattern for creating benchmark instances:
- `BenchmarkFactory.create(name, customConfig)` - Create benchmark instances
- Caches instances for performance
- Supports legacy configuration format
- Provides utility methods for querying benchmarks by tag, type, etc.

#### 4. **benchmark-renderer.js**
Handles UI rendering for individual benchmark sections:
- `BenchmarkRenderer` class manages one benchmark section
- Creates expandable headers, charts grid, score summaries
- Handles expand/collapse functionality
- Manages chart lifecycle

#### 5. **ui-manager.js**
Coordinates overall UI state and multiple renderers:
- `UIManager` class manages all benchmark renderers
- Handles filtering, search, expand/collapse all
- Manages application state
- Updates URL parameters

#### 6. **migration-adapter.js**
Provides backward compatibility:
- `MigrationAdapter` converts legacy config format to new format
- `BenchmarkApp` main application class
- Legacy API wrapper maintains `window.initAndRender()` compatibility

#### 7. **main.js**
New entry point that bridges old and new systems:
- Exports both legacy and modern APIs
- `window.initAndRender()` - Legacy API (backward compatible)
- `window.VortexBenchmarks` - Modern API with full access to classes

## Usage

### Legacy API (Backward Compatible)

The existing code in index.html continues to work unchanged:

```javascript
window.initAndRender([
  ["Clickbench", {
    hiddenDatasets: new Set(["datafusion:lance"]),
    renamedDatasets: { /* ... */ }
  }],
  ["TPC-H (NVMe) (SF=10)", { /* ... */ }]
]);
```

### Modern API

New applications can use the modular API directly:

```javascript
// Create a benchmark instance
const benchmark = VortexBenchmarks.createBenchmark('Clickbench', {
  hiddenDatasets: new Set(['custom-dataset'])
});

// Initialize application
const app = await VortexBenchmarks.initialize();

// Or create and configure manually
const app = new VortexBenchmarks.BenchmarkApp();
await app.initialize(customConfigs);
```

## Key Benefits

### 1. **Modularity**
Each component has a single responsibility and can be tested independently.

### 2. **Consistency**
All query benchmarks (Clickbench, TPC-H, TPC-DS, StatPopGen) share the same behavior through the `QueryBenchmark` class.

### 3. **Maintainability**
- Configuration is centralized in `benchmark-config.js`
- Adding new benchmarks only requires updating configuration
- Business logic is separated from UI rendering

### 4. **Extensibility**
Easy to add new benchmark types:
```javascript
class CustomBenchmark extends BaseBenchmark {
  calculateScore(benchSet) {
    // Custom scoring logic
  }
}
```

### 5. **No Breaking Changes**
The legacy API is fully preserved through the migration adapter, ensuring existing code continues to work.

## Configuration Structure

Benchmark configurations follow this structure:

```javascript
BENCHMARK_CONFIGS = {
  "Benchmark Name": {
    type: BenchmarkClass,        // Class to instantiate
    config: {
      queryType: "clickbench",   // For query benchmarks
      description: "...",         // Benchmark description
      tags: ["tag1", "tag2"],     // Category tags
      hiddenDatasets: new Set(), // Initially hidden datasets
      removedDatasets: new Set(), // Datasets to filter out
      renamedDatasets: {},        // Dataset name mappings
      keptCharts: [],            // Charts to keep (compression)
      // ... other type-specific config
    }
  }
}
```

## Testing

### Unit Tests
Test individual components:
- `test-modular.html` - Browser-based test suite
- `validate-consistency.js` - Node.js validation script

### Integration Testing
The system maintains full backward compatibility, so existing functionality serves as integration tests.

### Validation
- Configuration consistency is validated on load
- Numerical output is identical to the original system
- All hidden datasets (like lance) are properly configured

## Migration Path

1. **Phase 1 (Current)**: Parallel implementation
   - New modular system runs alongside old system
   - Full backward compatibility maintained
   - No breaking changes

2. **Phase 2 (Future)**: Gradual adoption
   - New features use modular API
   - Old code gradually migrated to new API
   - Legacy adapter remains for compatibility

3. **Phase 3 (Future)**: Full migration
   - All code uses modern API
   - Legacy adapter can be deprecated
   - Clean, fully modular codebase

## File Organization

```
benchmarks-website/
├── Core Modules
│   ├── benchmark-types.js      # Class definitions
│   ├── benchmark-config.js     # Configuration
│   ├── benchmark-factory.js    # Factory pattern
│   ├── benchmark-renderer.js   # UI rendering
│   ├── ui-manager.js          # UI coordination
│   ├── migration-adapter.js   # Backward compatibility
│   └── main.js                # Entry point
│
├── Existing Files (unchanged)
│   ├── code.js                # Original code
│   ├── chart-manager.js      # Chart management
│   ├── scoring.js            # Score calculation
│   ├── config.js             # Constants
│   └── ...other files
│
└── Tests
    ├── test-modular.html      # Browser tests
    └── validate-consistency.js # Validation script
```

## Common Tasks

### Adding a New Benchmark

1. Add configuration to `benchmark-config.js`:
```javascript
BENCHMARK_CONFIGS["My Benchmark"] = {
  type: QueryBenchmark,  // or appropriate type
  config: {
    description: "My benchmark description",
    tags: ["Queries (NVMe)"],
    hiddenDatasets: new Set(["slow-target"]),
    // ... other config
  }
};
```

2. The benchmark will automatically appear with correct behavior.

### Hiding a Dataset by Default

Update the configuration in `benchmark-config.js`:
```javascript
hiddenDatasets: new Set(["datafusion:lance", "another-dataset"])
```

### Customizing Chart Options

Override `getChartOptions()` in a custom benchmark class:
```javascript
class CustomBenchmark extends BaseBenchmark {
  getChartOptions() {
    return {
      scales: {
        y: {
          type: 'linear',  // Use linear instead of log scale
          // ... custom options
        }
      }
    };
  }
}
```

## Conclusion

The modular architecture provides a solid foundation for the Vortex Benchmarks website, ensuring:
- **Consistency** across all benchmarks
- **Maintainability** through clear separation of concerns
- **Extensibility** for future enhancements
- **Reliability** through preserved backward compatibility

The architecture is designed to scale with the project's needs while maintaining code quality and developer experience.