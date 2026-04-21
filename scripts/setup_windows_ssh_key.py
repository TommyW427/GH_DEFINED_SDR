#!/usr/bin/env python3
"""Install this machine's SSH public key on a Windows OpenSSH host."""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up passwordless SSH to a Windows OpenSSH host.")
    parser.add_argument("--host", required=True, help="Windows SSH target, e.g. tpdub@10.41.1.254")
    parser.add_argument("--key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
    return result


def ssh_base(host: str, port: int | None = None) -> list[str]:
    cmd = ["ssh"]
    if port is not None:
        cmd.extend(["-p", str(port)])
    cmd.append(host)
    return cmd


def ensure_key(key_path: Path, force_generate: bool) -> Path:
    pub_path = key_path.with_suffix(key_path.suffix + ".pub")
    if force_generate or not key_path.exists() or not pub_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        run(["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", ""])
    if not pub_path.exists():
        raise FileNotFoundError(str(pub_path))
    return pub_path


def main() -> int:
    args = parse_args()
    key_path = Path(args.key).expanduser()
    pub_path = ensure_key(key_path, args.force_generate)
    public_key = pub_path.read_text(encoding="utf-8").strip()
    if not public_key:
        raise RuntimeError(f"Public key is empty: {pub_path}")

    ps = f"""
$ErrorActionPreference = 'Stop'
$sshDir = Join-Path $HOME '.ssh'
$auth = Join-Path $sshDir 'authorized_keys'
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
if (-not (Test-Path -LiteralPath $auth)) {{
    New-Item -ItemType File -Force -Path $auth | Out-Null
}}
$key = @'
{public_key}
'@
$existing = Get-Content -LiteralPath $auth -ErrorAction SilentlyContinue
if ($existing -notcontains $key) {{
    Add-Content -LiteralPath $auth -Value $key
}}
icacls $sshDir /inheritance:r /grant:r "$env:USERNAME:(OI)(CI)F" | Out-Null
icacls $auth /inheritance:r /grant:r "$env:USERNAME:F" | Out-Null
Write-Host "installed"
"""
    encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
    print("Installing public key on Windows host. You may be prompted for the Windows password once.")
    run([*ssh_base(args.host, args.ssh_port), f"powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"])

    test_cmd = [*ssh_base(args.host, args.ssh_port)]
    test_cmd.insert(1, "-o")
    test_cmd.insert(2, "BatchMode=yes")
    test_cmd.append("echo ok")
    print("Testing passwordless SSH...")
    result = run(test_cmd, check=False)
    if result.returncode != 0:
        print("Passwordless SSH test failed. Check Windows OpenSSH AuthorizedKeysFile and file permissions.", file=sys.stderr)
        return result.returncode
    print("Passwordless SSH is configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
