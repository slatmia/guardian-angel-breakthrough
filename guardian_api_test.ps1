param(
    [int]$Port = 11436,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GUARDIAN DECISION CAPSULE – API TEST" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Target: http://$HostName`:$Port" -ForegroundColor DarkCyan
Write-Host ""

# -------------------------------
# 1) Health check
# -------------------------------
$healthUrl = "http://$HostName`:$Port/health"
Write-Host "[1] Health check: $healthUrl" -ForegroundColor Yellow

try {
    $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5
    Write-Host "  ✓ Health OK" -ForegroundColor Green
    $health | Format-List
}
catch {
    Write-Host "  ✗ Health FAILED: $($_.Exception.Message)" -ForegroundColor Red
    return
}

Write-Host ""
# -------------------------------
# 2) Analyze sample text
# -------------------------------
$analyzeUrl = "http://$HostName`:$Port/guardian/analyze"
Write-Host "[2] Analyze endpoint: $analyzeUrl" -ForegroundColor Yellow

$sampleText = "I hate this stupid code! DELETE FROM users;"

$body = @{
    text = $sampleText
} | ConvertTo-Json

Write-Host "  Sending sample text:" -ForegroundColor DarkYellow
Write-Host "    $sampleText" -ForegroundColor DarkYellow
Write-Host ""

try {
    $response = Invoke-RestMethod `
        -Uri $analyzeUrl `
        -Method Post `
        -Body $body `
        -ContentType "application/json" `
        -TimeoutSec 10

    Write-Host "  ✓ Analyze CALL OK" -ForegroundColor Green
    Write-Host ""
    Write-Host "===== RAW JSON RESPONSE =====" -ForegroundColor DarkCyan
    $response | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "  ✗ Analyze FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GUARDIAN API TEST COMPLETE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan