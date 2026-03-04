.PHONY: serve clean-web

web/pkg: src/**/*.rs Cargo.toml
	wasm-pack build --target web --no-default-features --out-dir web/pkg

serve: web/pkg
	@echo "Open http://localhost:8000"
	cd web && python3 -m http.server

clean-web:
	rm -rf web/pkg
