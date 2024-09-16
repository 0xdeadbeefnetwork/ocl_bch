# Bitcoin Cash (BCH) Wallet Generator with OpenCL

This project generates Bitcoin Cash (BCH) wallets using **OpenCL** for GPU-accelerated entropy gathering and SHA-256 hashing. The project outputs both **CashAddr** (modern) and **Legacy** (starting with `1`) Bitcoin Cash addresses from the same private key. The generated private key is also provided in **WIF (Wallet Import Format)** for easy import into other wallet software.

## Features

- **OpenCL-powered entropy generation**: Leverages the GPU for cryptographic randomness.
- **GPU-accelerated SHA-256 hashing**: Performs fast SHA-256 hashing on the GPU.
- **Bitcoin Cash address generation**: Outputs both CashAddr and Legacy Bitcoin Cash addresses.
- **WIF support**: Exports the private key in Wallet Import Format for importing into other wallets.
- **Cross-platform**: Compatible with systems that support Python and OpenCL.

## Prerequisites

- **Python 3.6+**
- **PyOpenCL** for OpenCL bindings in Python.
- A GPU with OpenCL support.
- **bitcash** for Bitcoin Cash key generation and address formatting.
- OpenCL-compatible GPU drivers installed on your system.

### Required Python Libraries

You can install the required libraries via `pip`:

```bash
pip install pyopencl bitcash numpy
