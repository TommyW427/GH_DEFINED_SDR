#!/usr/bin/env python3
"""Print remote SoapySDR diagnostics and verify that a device can be opened."""

from __future__ import annotations

import argparse
import json
import sys

import SoapySDR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote SoapySDR preflight.")
    parser.add_argument("--device-args", required=True)
    parser.add_argument("--direction", default="TX")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("SoapySDR API:", SoapySDR.getAPIVersion())
    try:
        print("SoapySDR root:", SoapySDR.getRootPath())
    except Exception as exc:
        print("SoapySDR root unavailable:", exc)
    try:
        print("SoapySDR search paths:", SoapySDR.listSearchPaths())
    except Exception as exc:
        print("SoapySDR search paths unavailable:", exc)
    try:
        print("SoapySDR modules:", SoapySDR.listModules())
    except Exception as exc:
        print("SoapySDR modules unavailable:", exc)

    devices = SoapySDR.Device.enumerate()
    print("SoapySDR devices:", json.dumps([dict(device) for device in devices], indent=2))
    try:
        dev = SoapySDR.Device(args.device_args)
        print(f"Remote {args.direction} device open OK:", args.device_args)
        print("Remote hardware:", dev.getHardwareKey())
    except Exception as exc:
        print(f"Remote {args.direction} device open FAILED:", args.device_args)
        print(type(exc).__name__ + ":", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
