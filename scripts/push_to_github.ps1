param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteUrl
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$existing = git remote 2>$null
if ($existing -contains "origin") {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

git branch -M main
git push -u origin main
git push origin --tags
Write-Host "SafeGuard main branch and milestone tags synchronized." -ForegroundColor Green
