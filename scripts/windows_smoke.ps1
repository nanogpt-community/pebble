param(
    [string]$Binary = "target/debug/pebble.exe"
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path $Binary).Path
$configHome = Join-Path $env:TEMP ("pebble-windows-smoke-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $configHome | Out-Null

try {
    $env:PEBBLE_CONFIG_HOME = $configHome
    $env:NO_COLOR = "1"
    $output = (& $resolved --help 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "pebble --help exited with $LASTEXITCODE"
    }
    if ($output.Contains([char]27)) {
        throw "NO_COLOR output contained an ANSI escape"
    }
    if (-not $output.Contains("pebble doctor providers")) {
        throw "help output did not include provider diagnostics"
    }

    Remove-Item Env:NO_COLOR
    $env:TERM = "dumb"
    $output = (& $resolved --version 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $output.Contains([char]27)) {
        throw "TERM=dumb version smoke failed"
    }
}
finally {
    Remove-Item -Recurse -Force $configHome -ErrorAction SilentlyContinue
}

Write-Host "Windows redirected-terminal smoke checks passed"
