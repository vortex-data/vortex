# Vortex WASM Test

Integration test for Vortex library compiled to WebAssembly.

## Building

### Prerequisites

1. Install the WASM target:

```bash
rustup target add wasm32-unknown-unknown
```

2. Install wasm-pack:

```bash
cargo install wasm-pack
```

### Build Steps

1. Navigate to the wasm-test directory:

```bash
cd wasm-test
```

2. Build for web:

```bash
wasm-pack build --target web
```

This creates the `pkg/` directory with JS bindings automatically.

## Testing

### In Browser

1. Start a local web server (required for WASM loading):

Using Python:

```bash
python3 -m http.server 8000
```

2. Open your browser to `http://localhost:8000`

3. Click the test buttons:
   - **Test Basic Function** - Tests simple `add()` function.
   - **Get Version** - Gets version string.
   - **Test Vortex Arrays** - Tests PrimitiveArray, compute operations, and encodings.
   - **Test Compression** - Tests BtrBlocksCompressor compression.
   - **Test Array Types** - Tests different array types (ConstantArray, StructArray, etc.).
   - **Test Compute Operations** - Tests comparison operations (>, >=, ==).

Console output from the WASM module will be displayed in the output area.

### Headless Tests (wasm-bindgen)

Run wasm-bindgen tests in headless Chrome:

```bash
wasm-pack test --headless --chrome
```

Or Firefox:

```bash
wasm-pack test --headless --firefox
```

### WASI Tests (Wasmer)

1. Install the WASI target and Wasmer:

```bash
rustup target add wasm32-wasip1
curl https://get.wasmer.io -sSfL | sh
```

2. Build and run:

```bash
cargo build --target wasm32-wasip1
wasmer run ./target/wasm32-wasip1/debug/wasm-test.wasm
```

## Project Structure

- `src/lib.rs` - WASM library with wasm-bindgen exports.
- `src/main.rs` - WASI binary for integration testing via Wasmer.
- `index.html` - Browser test page.
- `pkg/` - Generated JS bindings (created by wasm-pack).
