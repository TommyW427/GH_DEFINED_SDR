param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [Parameter(Mandatory = $true)][string]$Capture,
    [Parameter(Mandatory = $true)][string]$CondaExe,
    [Parameter(Mandatory = $false)][string]$CondaEnv = "",
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true)][string]$ArgvJsonBase64
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-CondaExe {
    param([string]$Requested)

    if ($Requested -and (Test-Path -LiteralPath $Requested)) {
        return $Requested
    }
    if ($Requested -and (Get-Command $Requested -ErrorAction SilentlyContinue)) {
        return $Requested
    }

    $candidates = @(
        "$env:USERPROFILE/radioconda/condabin/conda.bat",
        "$env:USERPROFILE/radioconda/Scripts/conda.exe",
        "$env:USERPROFILE/miniconda3/Scripts/conda.exe",
        "$env:USERPROFILE/anaconda3/Scripts/conda.exe",
        "$env:USERPROFILE/miniforge3/Scripts/conda.exe",
        "$env:USERPROFILE/mambaforge/Scripts/conda.exe",
        "C:/radioconda/condabin/conda.bat",
        "C:/radioconda/Scripts/conda.exe",
        "C:/ProgramData/miniconda3/Scripts/conda.exe",
        "C:/ProgramData/anaconda3/Scripts/conda.exe",
        "C:/ProgramData/miniforge3/Scripts/conda.exe",
        "C:/tools/miniconda3/Scripts/conda.exe"
    )

    foreach ($candidate in $candidates) {
        $expanded = [Environment]::ExpandEnvironmentVariables($candidate)
        if (Test-Path -LiteralPath $expanded) {
            return $expanded
        }
    }

    throw "Could not find conda. Pass --remote-conda-exe with the full Windows path to conda.exe or conda.bat."
}

New-Item -ItemType Directory -Force -Path $Capture | Out-Null
Set-Location $Repo

if ($CondaEnv) {
    $resolvedConda = Resolve-CondaExe -Requested $CondaExe
    if ($resolvedConda.ToLower().EndsWith(".bat")) {
        $condaHook = cmd.exe /d /c "`"$resolvedConda`" shell.powershell hook"
    } else {
        $condaHook = & $resolvedConda shell.powershell hook
    }
    $condaHook | Out-String | Invoke-Expression
    conda activate $CondaEnv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate remote conda environment: $CondaEnv"
    }
    python -c "import gnuradio"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote conda environment does not provide gnuradio: $CondaEnv"
    }
}

$env:PYTHONPATH = $PythonPath
$argvJson = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($ArgvJsonBase64))
$decodedArgv = $argvJson | ConvertFrom-Json
$remoteArgv = @()
foreach ($item in $decodedArgv) {
    $remoteArgv += [string]$item
}
if ($remoteArgv.Count -lt 1) {
    throw "Remote argv is empty."
}

$program = [string]($remoteArgv[0])
$programArgs = @()
if ($remoteArgv.Count -gt 1) {
    $programArgs = @($remoteArgv[1..($remoteArgv.Count - 1)] | ForEach-Object { [string]($_) })
}

& $program @programArgs
exit $LASTEXITCODE
