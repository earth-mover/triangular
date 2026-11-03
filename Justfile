# Task runner. `just` lists the recipes.

default:
    @just --list

# Build the extension into the uv environment.
build:
    uv sync --locked

check:
    cargo check --all-targets --all-features

lint:
    cargo clippy --all-targets --all-features -- -D warnings
    uv run --locked ruff check .
    uv run --locked pyright

format *args:
    cargo fmt --all {{ args }}
    uv run --locked ruff format {{ args }} .

alias fmt := format

test: test-rust test-py

# Rust unit tests. The python feature links against libpython, so it stays off.
test-rust *args:
    cargo test --no-default-features {{ args }}

test-py *args:
    uv run --locked pytest -m "not benchmark and not slow" {{ args }}

bench *args:
    uv run --locked pytest -m benchmark {{ args }}

check-deps:
    cargo deny check

ci:
    just format --check
    just lint
    just test

check-all: ci
    just check-deps
