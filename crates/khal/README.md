# khal - A hardware abstraction for compute shaders

The **khal** (Kompute Hardware Abstraction Layer) library provides abstractions for running shaders on any platform.

> **Warning**
> **khal** is still very incomplete and under heavy development and is lacking a lot of features and backends.

### Other features

**khal** also provides utilities for:
- Writing device-side gpu code in a backend-agnostic way, with the ability to reuse the same code on multiple backend
  even within the same executable.
- Generating boilerplate and helper functions for loading a shader from Rust and launching its compute pipeline.
