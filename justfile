# List all commands
_default:
  @just --list --unsorted

# Download mnist dataset
download:
	#!/usr/bin/env bash
	mkdir mnist
	cd mnist
	for file in train-images-idx3 train-labels-idx1 t10k-images-idx3 t10k-labels-idx1; do
	echo http://yann.lecun.com/exdb/mnist/"$file"-ubyte.gz
		curl -o - http://yann.lecun.com/exdb/mnist/"$file"-ubyte.gz |  gunzip --stdout > "$file"-ubyte
	done

# Initialize build directory
init mode:
	@meson setup --buildtype {{mode}} build/{{mode}} 

# Build project in mode
build mode:
	@[ -d build/{{mode}} ] || just init {{mode}}; ninja -C build/{{mode}}

# Compile and run debug build
debug *args:
	@just build debugoptimized
	@build/debugoptimized/neural {{args}}

# Run debug build in gdb
gdb *args:
	@just build debugoptimized
	@gdb -ex=r --args build/debugoptimized/neural {{args}}

# Compile and run release build
release *args:
	@just build release
	@build/release/neural {{args}}

# Compile and run tests
[no-exit-message]
test *args:
	@just build debugoptimized
	@build/debugoptimized/neural-test {{args}}

# Generate backprop test using PyTorch as reference
generate-backprop-test:
	@nix develop .#scripts -c python scripts/generate_test.py

# Delete all build artifacts
clean:
	rm -rf build
