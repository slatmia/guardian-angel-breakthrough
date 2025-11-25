# ================================================================================
# 🐝 GUARDIAN SWARM ROUTER - INTELLIGENT MULTI-AGENT ORCHESTRATION
# ================================================================================
# Routes prompts to the best specialist: Emotion, Security, or Quality
# ================================================================================

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Prompt,
    
    [Parameter(Mandatory=$false)]
    [switch]$Validate,
    
    [Parameter(Mandatory=$false)]
    [switch]$VerboseOutput
)

Write-Host "`n🐝 Guardian Swarm Router - v1.0" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor DarkGray

# === CONFIGURATION ===
$EmotionModel = "gemma3-emotion-enhanced:latest"
$SecurityModel = "guardian-security:v1.0"
$QualityModel = "guardian-quality:v1.0"
$GuardianModel = "guardian-angel:breakthrough-v2"
$ClassifierModel = "gemma:2b-instruct-q4_0"

# === PHASE 1: INTENT CLASSIFICATION ===
Write-Host "`n🎯 Phase 1: Classifying intent..." -ForegroundColor Yellow

# Pre-classification: Keyword-based routing (bypass unreliable 2B model)
$intentKeywords = @{
    "C" = @("refactor", "SOLID", "SRP", "OCP", "LSP", "ISP", "DIP", "design pattern", "clean code", 
            "architecture", "complexity", "maintainability", "code quality", "code smell", "technical debt")
    "B" = @("security", "vulnerability", "exploit", "SQL injection", "XSS", "CSRF", "authentication", 
            "authorization", "encryption", "crypto", "secure", "attack", "breach")
    "A" = @("burnout", "stressed", "frustrated", "overwhelmed", "tired", "help me feel", 
            "encourage", "support", "motivation", "exhausted", "struggling", "anxious")
}

$detectedIntent = $null
$promptLower = $Prompt.ToLower()

foreach ($category in @("C", "B", "A")) {  # Check Quality first, then Security, then Emotion
    foreach ($keyword in $intentKeywords[$category]) {
        if ($promptLower -contains $keyword -or $promptLower -match [regex]::Escape($keyword)) {
            $detectedIntent = $category
            Write-Host "   🔍 Keyword match: '$keyword' → Category $category" -ForegroundColor Green
            break
        }
    }
    if ($detectedIntent) { break }
}

# Fallback to AI classification if no keywords matched
if (-not $detectedIntent) {
    Write-Host "   🤖 No keyword match, using AI classifier..." -ForegroundColor Yellow
    
    $classifierPrompt = @"
Classify this prompt into ONE category:

A) EMOTION - feelings, stress, burnout, overwhelmed, frustrated, tired, encouragement, motivation, support
   Keywords: burnout, stressed, frustrated, overwhelmed, tired, help me feel, encourage, support

B) SECURITY - vulnerabilities, exploits, authentication, authorization, encryption, SQL injection, XSS, CSRF
   Keywords: security, vulnerability, exploit, auth, encryption, injection, XSS, CSRF, crypto, secure

C) QUALITY - code architecture, refactoring, SOLID principles, design patterns, clean code, maintainability
   Keywords: refactor, SOLID, SRP, OCP, DIP, architecture, design pattern, clean code, complexity, quality, maintainability

Prompt: $Prompt

Respond with ONLY A, B, or C. No explanation.
"@

$startClassify = Get-Date
$intentResponse = ollama run $ClassifierModel $classifierPrompt --temperature 0 2>&1 | Out-String
$endClassify = Get-Date
$classifyTime = ($endClassify - $startClassify).TotalSeconds

# Extract intent
$intent = $intentResponse.Trim().ToUpper()
if ($intent -match '[ABC]') {
    $intent = ($intent | Select-String -Pattern '[ABC]' -AllMatches).Matches[0].Value
} else {
    $intent = "C"  # Default to Quality
}

    $detectedIntent = $intent
    Write-Host "   AI Classification: $detectedIntent" -ForegroundColor Green
    Write-Host "   Classification time: $([math]::Round($classifyTime, 2))s" -ForegroundColor Gray
} else {
    $classifyTime = 0
}

$intent = $detectedIntent

# Map to agent
$agentMap = @{
    "A" = @{name="Emotion"; model=$EmotionModel; description="Empathy & burnout support"}
    "B" = @{name="Security"; model=$SecurityModel; description="Auth, encryption, vulnerabilities"}
    "C" = @{name="Quality"; model=$QualityModel; description="SOLID, refactoring, patterns"}
}

$selectedAgent = $agentMap[$intent]

Write-Host "   Intent: $intent → $($selectedAgent.name)" -ForegroundColor Green
Write-Host "   Model: $($selectedAgent.model)" -ForegroundColor Gray
Write-Host "   Classification time: $([math]::Round($classifyTime, 2))s" -ForegroundColor Gray

# === PHASE 2: GENERATE WITH SPECIALIST ===
Write-Host "`n⚡ Phase 2: Generating response..." -ForegroundColor Yellow
Write-Host "   Agent: $($selectedAgent.name) ($($selectedAgent.description))" -ForegroundColor Gray

$startGenerate = Get-Date
$response = ollama run $selectedAgent.model $Prompt 2>&1 | Out-String
$endGenerate = Get-Date
$generateTime = ($endGenerate - $startGenerate).TotalSeconds

Write-Host "   Generated: $($response.Length) characters" -ForegroundColor Green
Write-Host "   Generation time: $([math]::Round($generateTime, 2))s" -ForegroundColor Gray

# === PHASE 3: GUARDIAN VALIDATION (OPTIONAL) ===
$isSafe = $true
$validationTime = 0

if ($Validate) {
    Write-Host "`n🛡️  Phase 3: Guardian validation..." -ForegroundColor Yellow
    
    $validationPrompt = @"
Analyze this response for Guardian Angel violations:
- Harmful content
- Unsafe code practices
- Unethical suggestions
- Burnout-inducing tone

Response: $response

Reply with ONLY: SAFE or VIOLATION
"@
    
    $startValidate = Get-Date
    $validationResponse = ollama run $GuardianModel $validationPrompt --temperature 0 2>&1 | Out-String
    $endValidate = Get-Date
    $validationTime = ($endValidate - $startValidate).TotalSeconds
    
    $isSafe = $validationResponse -notmatch "VIOLATION"
    
    if ($isSafe) {
        Write-Host "   Status: ✅ SAFE" -ForegroundColor Green
    } else {
        Write-Host "   Status: ⚠️  VIOLATION DETECTED" -ForegroundColor Red
    }
    Write-Host "   Validation time: $([math]::Round($validationTime, 2))s" -ForegroundColor Gray
}

# === RESULTS ===
$totalTime = $classifyTime + $generateTime + $validationTime

Write-Host "`n" -NoNewline
Write-Host "="*80 -ForegroundColor DarkGray
Write-Host "📊 ROUTING SUMMARY" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor DarkGray
Write-Host ""
Write-Host "   Agent Selected: $($selectedAgent.name)" -ForegroundColor White
Write-Host "   Model Used: $($selectedAgent.model)" -ForegroundColor Gray
Write-Host "   Safety Status: $(if($isSafe){'✅ Safe'}else{'⚠️  Flagged'})" -ForegroundColor $(if($isSafe){'Green'}else{'Red'})
Write-Host "   Total Latency: $([math]::Round($totalTime, 2))s" -ForegroundColor Yellow
Write-Host "      ├─ Classification: $([math]::Round($classifyTime, 2))s" -ForegroundColor Gray
Write-Host "      ├─ Generation: $([math]::Round($generateTime, 2))s" -ForegroundColor Gray
if ($Validate) {
    Write-Host "      └─ Validation: $([math]::Round($validationTime, 2))s" -ForegroundColor Gray
}
Write-Host ""

# === OUTPUT RESPONSE ===
Write-Host "="*80 -ForegroundColor DarkGray
Write-Host "💬 RESPONSE FROM $($selectedAgent.name.ToUpper())" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor DarkGray
Write-Host ""
Write-Host $response
Write-Host ""

# === VERBOSE MODE ===
if ($VerboseOutput) {
    Write-Host "="*80 -ForegroundColor DarkGray
    Write-Host "🔍 VERBOSE DEBUG INFO" -ForegroundColor Yellow
    Write-Host "="*80 -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Classifier Prompt:" -ForegroundColor Gray
    Write-Host $classifierPrompt
    Write-Host ""
    Write-Host "Raw Intent Response:" -ForegroundColor Gray
    Write-Host $intentResponse
    if ($Validate) {
        Write-Host ""
        Write-Host "Validation Response:" -ForegroundColor Gray
        Write-Host $validationResponse
    }
}
