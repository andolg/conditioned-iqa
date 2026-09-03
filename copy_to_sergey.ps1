[CmdletBinding()]
param(
    # Defaults to the directory from which this script is invoked.
    [string]$SourcePath = (Get-Location).Path,

    [string]$HostName = '10.24.1.21',
    [string]$UserName = 'sergey',
    [string]$RemoteDirectory = '~/conditioned-iqa/28d_evs/project-1-conditioned-iqa'
)

$ErrorActionPreference = 'Stop'

$SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
    throw "Source directory does not exist: $SourcePath"
}

$sourceName = Split-Path -Leaf $SourcePath
$remoteParent = Split-Path -Parent $RemoteDirectory
$remoteTarget = "$UserName@$HostName"

foreach ($command in 'ssh', 'scp') {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "'$command' was not found. Install the Windows OpenSSH Client and try again."
    }
}

Write-Host "Creating $remoteParent on $remoteTarget ..."
& ssh $remoteTarget "mkdir -p '$remoteParent'"
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the remote directory."
}

Write-Host "Copying $SourcePath to $remoteTarget`:$remoteParent/ ..."
& scp -r -- $SourcePath "$remoteTarget`:$remoteParent/"
if ($LASTEXITCODE -ne 0) {
    throw "Copy failed."
}

Write-Host "Done: $remoteTarget`:$RemoteDirectory"
