param(
    [Parameter(Mandatory=$true)]
    [string]$Prompt,
    
    [int]$Timeout = 60,
    
    [switch]$Validate,
    
    [switch]$VerboseOutput
)

# === CACHE SETUP ===
$cacheFile = "$env:USERPROFILE\.guardian_swarm_cache.json"
$cache = if (Test-Path $cacheFile) { 
    Get-Content $cacheFile -Raw | ConvertFrom-Json -AsHashtable 
} else { 
    @{} 
}

# === KEYWORD CLASSIFICATION (0ms) ===
$intentKeywords = @{
    "C" = @("refactor", "SOLID", "SRP", "OCP", "LSP", "ISP", "DIP", "design pattern", "clean code", 
            "architecture", "complexity", "maintainability", "code smell", "testability", "coupling")
    "B" = @("SQL injection", "XSS", "CSRF", "authentication", "authorization", "encryption", 
            "vulnerability", "exploit", "crypto", "secure", "password", "token", "session")
    "A" = @("burnout", "stressed", "frustrated", "overwhelmed", "tired", "help me feel", 
            "encourage", "support", "motivation", "feeling", "exhausted", "anxious")
}

# === MEASURE CLASSIFICATION TIME ===
$classifyStart = Get-Date
$detectedIntent = $null
$matchedKeyword = $null
$promptLower = $Prompt.ToLower()

# Check keywords in priority order (Quality -> Security -> Emotion)
foreach ($category in @("C", "B", "A")) {
    foreach ($keyword in $intentKeywords[$category]) {
        if ($promptLower -match [regex]::Escape($keyword.ToLower())) {
            $detectedIntent = $category
            $matchedKeyword = $keyword
            break
        }
    }
    if ($detectedIntent) { break }
}

$classifyEnd = Get-Date
$classifyTime = ($classifyEnd - $classifyStart).TotalSeconds

# === CACHE LOGIC ===
$cacheKey = $Prompt.Substring(0, [Math]::Min(100, $Prompt.Length))
$cacheHit = $false

if ($cache.ContainsKey($cacheKey)) {
    $cachedIntent = $cache[$cacheKey]
    if ($VerboseOutput) {
        Write-Host "⚡ CACHE HIT! Previous routing: $cachedIntent" -ForegroundColor Green
    }
    $cacheHit = $true
} else {
    $cache[$cacheKey] = $detectedIntent
    $cache | ConvertTo-Json | Set-Content $cacheFile -Encoding UTF8
    if ($VerboseOutput) {
        Write-Host "💾 Cache updated with new routing" -ForegroundColor Yellow
    }
}

# === DEFAULT TO QUALITY IF NO MATCH ===
if (-not $detectedIntent) {
    $detectedIntent = "C"
    $matchedKeyword = "default"
}

# === MAP TO AGENT ===
$agentMap = @{
    "A" = @{Name = "Emotion"; Model = "gemma3-emotion-enhanced:latest"; Description = "Empathy & burnout support"}
    "B" = @{Name = "Security"; Model = "guardian-security:v1.0"; Description = "Vulnerability detection"}
    "C" = @{Name = "Quality"; Model = "guardian-quality:v1.0"; Description = "SOLID compliance & refactoring"}
}

$selectedAgent = $agentMap[$detectedIntent]

# === DISPLAY HEADER ===
Write-Host "`n$("=" * 80)" -ForegroundColor Cyan
Write-Host "🐝 Guardian Swarm Router - v2.0 (OPTIMIZED)" -ForegroundColor Cyan
Write-Host "$("=" * 80)`n" -ForegroundColor Cyan

if ($VerboseOutput) {
    Write-Host "🎯 Classification Method: $(if ($matchedKeyword -ne 'default') { 'Keyword Match (0ms)' } else { 'Default Fallback' })" -ForegroundColor Yellow
    if ($matchedKeyword -ne "default") {
        Write-Host "🔍 Keyword matched: '$matchedKeyword' → Category $detectedIntent" -ForegroundColor Green
    }
    Write-Host "📊 Cache Status: $(if ($cacheHit) { '✅ HIT' } else { '❌ MISS' })" -ForegroundColor $(if ($cacheHit) { 'Green' } else { 'Yellow' })
    Write-Host "⚡ Agent Selected: $($selectedAgent.Name)" -ForegroundColor Cyan
    Write-Host "🤖 Model: $($selectedAgent.Model)`n" -ForegroundColor Gray
}

Write-Host "🎯 Intent: $detectedIntent → $($selectedAgent.Name)" -ForegroundColor Yellow
Write-Host "   Model: $($selectedAgent.Model)" -ForegroundColor Gray
Write-Host "   Classification time: $([math]::Round($classifyTime, 3))s`n" -ForegroundColor Gray

# === GENERATE RESPONSE WITH STREAMING ===
Write-Host "⚡ Phase 2: Generating response..." -ForegroundColor Yellow
Write-Host "   Agent: $($selectedAgent.Name) ($($selectedAgent.Description))`n" -ForegroundColor Gray

$generateStart = Get-Date

Write-Host "$("=" * 80)" -ForegroundColor DarkGray
Write-Host "💬 RESPONSE FROM $($selectedAgent.Name.ToUpper())" -ForegroundColor Cyan
Write-Host "$("=" * 80)`n" -ForegroundColor DarkGray

# Generate with streaming
$response = ollama run $selectedAgent.Model $Prompt

$generateEnd = Get-Date
$generateTime = ($generateEnd - $generateStart).TotalSeconds

# === OPTIONAL GUARDIAN VALIDATION ===
$validationStatus = "✅ Safe (no validation requested)"
$validationTime = 0

if ($Validate) {
    Write-Host "`n$("=" * 80)" -ForegroundColor Yellow
    Write-Host "🛡️  Phase 3: Guardian Angel Validation..." -ForegroundColor Yellow
    Write-Host "$("=" * 80)`n" -ForegroundColor Yellow
    
    $validateStart = Get-Date
    $validationPrompt = "Does this response violate Guardian Angel principles? Respond SAFE or VIOLATION: $response"
    
    $validationResult = ollama run guardian-angel:breakthrough-v2 $validationPrompt --stream:$false
    
    $validateEnd = Get-Date
    $validationTime = ($validateEnd - $validateStart).TotalSeconds
    
    if ($validationResult -match "VIOLATION") {
        $validationStatus = "⚠️  WARNING: Potential violation detected"
        Write-Host $validationStatus -ForegroundColor Red
    } else {
        $validationStatus = "✅ Safe (validated by Guardian Angel)"
        Write-Host $validationStatus -ForegroundColor Green
    }
}

# === SUMMARY ===
$totalTime = $classifyTime + $generateTime + $validationTime

Write-Host "`n$("=" * 80)" -ForegroundColor Cyan
Write-Host "📊 ROUTING SUMMARY" -ForegroundColor Cyan
Write-Host "$("=" * 80)`n" -ForegroundColor Cyan

Write-Host "   Agent Selected: $($selectedAgent.Name)" -ForegroundColor Yellow
Write-Host "   Model Used: $($selectedAgent.Model)" -ForegroundColor Gray
Write-Host "   Classification: $(if ($matchedKeyword -ne 'default') { "Keyword: '$matchedKeyword'" } else { 'Default fallback' })" -ForegroundColor Gray
Write-Host "   Cache Status: $(if ($cacheHit) { 'HIT ⚡' } else { 'MISS 💾' })" -ForegroundColor $(if ($cacheHit) { 'Green' } else { 'Yellow' })
Write-Host "   Safety Status: $validationStatus" -ForegroundColor Gray
Write-Host "   Total Latency: $([math]::Round($totalTime, 2))s" -ForegroundColor Yellow
Write-Host "      ├─ Classification: $([math]::Round($classifyTime, 3))s" -ForegroundColor Gray
Write-Host "      ├─ Generation: $([math]::Round($generateTime, 2))s" -ForegroundColor Gray
if ($Validate) {
    Write-Host "      └─ Validation: $([math]::Round($validationTime, 2))s" -ForegroundColor Gray
}

Write-Host "`n$("=" * 80)`n" -ForegroundColor Cyan
