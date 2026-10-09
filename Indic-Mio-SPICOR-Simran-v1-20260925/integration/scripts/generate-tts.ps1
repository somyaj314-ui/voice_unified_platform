[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Text,

    [string]$TextFile,

    [ValidateSet('Main', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'Custom')]
    [string]$Profile = 'Main',

    [ValidateSet('main', 'base', 'r1', 'r2', 'r3', 'clone', 'clone-indian',
        'simran_polished_soft', 'simran_polite_short_0001',
        'simran_polite_short_0002', 'simran_polite_short_0003',
        'simran_clone_primary', 'simran_clone_indian',
        'bunty_punchy_a', 'bunty_punchy_b', 'bunty_punchy_c')]
    [string]$Reference = 'main',

    [Nullable[double]]$Temperature,
    [Nullable[double]]$TopP,
    [Nullable[double]]$RepetitionPenalty,
    [Nullable[int]]$MaxTokens,
    [Nullable[int]]$Seed,

    [ValidateSet('neutral', 'active', 'happy', 'sad', 'angry', 'whisper',
        'enunciated', 'confused', 'disgust', 'fear', 'surprise')]
    [string]$Emotion,

    [ValidateSet('English', 'Hindi', 'Hinglish', 'Auto')]
    [string]$Language = 'English',

    [string]$Output,
    [string]$Server = 'http://127.0.0.1:8080',
    [switch]$NoCache,
    [switch]$ListProfiles
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$Profiles = @{
    Main = @{ Name = 'Approved Simran MERI'; Temperature = 0.55; TopP = 0.87; Repetition = 1.04; MaxTokens = 650; Seed = 42; Emotion = 'neutral' }
    P1 = @{ Name = 'Conservative'; Temperature = 0.20; TopP = 0.65; Repetition = 1.05; MaxTokens = 300; Seed = 42; Emotion = 'neutral' }
    P2 = @{ Name = 'Stable';       Temperature = 0.35; TopP = 0.75; Repetition = 1.03; MaxTokens = 300; Seed = 42; Emotion = 'neutral' }
    P3 = @{ Name = 'Approved';     Temperature = 0.55; TopP = 0.87; Repetition = 1.04; MaxTokens = 650; Seed = 42; Emotion = 'neutral' }
    P4 = @{ Name = 'Natural';      Temperature = 0.62; TopP = 0.90; Repetition = 1.03; MaxTokens = 650; Seed = 42; Emotion = 'neutral' }
    P5 = @{ Name = 'Expressive';   Temperature = 0.75; TopP = 0.95; Repetition = 1.02; MaxTokens = 650; Seed = 42; Emotion = 'neutral' }
    P6 = @{ Name = 'Enunciated';   Temperature = 0.45; TopP = 0.80; Repetition = 1.04; MaxTokens = 650; Seed = 42; Emotion = 'enunciated' }
    P7 = @{ Name = 'Happy';        Temperature = 0.45; TopP = 0.82; Repetition = 1.04; MaxTokens = 650; Seed = 42; Emotion = 'happy' }
    Custom = @{ Name = 'Custom';   Temperature = 0.35; TopP = 0.75; Repetition = 1.03; MaxTokens = 300; Seed = 42; Emotion = 'neutral' }
}

$References = @{
    main = 'simran_polished_soft'
    base = 'simran_polished_soft'
    r1 = 'simran_polite_short_0001'
    r2 = 'simran_polite_short_0002'
    r3 = 'simran_polite_short_0003'
    clone = 'simran_clone_primary'
    'clone-indian' = 'simran_clone_indian'
    simran_polished_soft = 'simran_polished_soft'
    simran_polite_short_0001 = 'simran_polite_short_0001'
    simran_polite_short_0002 = 'simran_polite_short_0002'
    simran_polite_short_0003 = 'simran_polite_short_0003'
    simran_clone_primary = 'simran_clone_primary'
    simran_clone_indian = 'simran_clone_indian'
    bunty_punchy_a = 'bunty_punchy_a'
    bunty_punchy_b = 'bunty_punchy_b'
    bunty_punchy_c = 'bunty_punchy_c'
}

if ($ListProfiles) {
    $Profiles.GetEnumerator() | Sort-Object Name | ForEach-Object {
        [pscustomobject]@{
            Profile = $_.Key
            Style = $_.Value.Name
            Temperature = $_.Value.Temperature
            TopP = $_.Value.TopP
            RepetitionPenalty = $_.Value.Repetition
            MaxTokens = $_.Value.MaxTokens
            Seed = $_.Value.Seed
            Emotion = $_.Value.Emotion
        }
    } | Format-Table -AutoSize
    exit 0
}

if ($Text -and $TextFile) {
    throw 'Use either -Text or -TextFile, not both.'
}

$Lines = @()
if ($TextFile) {
    $resolvedTextFile = (Resolve-Path -LiteralPath $TextFile).Path
    $Lines = @(Get-Content -LiteralPath $resolvedTextFile -Encoding UTF8 |
        ForEach-Object { $_.Trim() } | Where-Object { $_ })
} elseif ($Text) {
    $Lines = @($Text.Trim())
} else {
    $entered = Read-Host 'Enter the sentence to generate'
    if ($entered.Trim()) { $Lines = @($entered.Trim()) }
}

if ($Lines.Count -eq 0) {
    throw 'No non-empty text was provided.'
}

try {
    $health = Invoke-RestMethod -Uri "$($Server.TrimEnd('/'))/health" -TimeoutSec 5
    if ($health.gateway -ne 'ready' -or $health.backend -ne 'ready') {
        throw "Gateway/backend is not ready: $($health | ConvertTo-Json -Compress)"
    }
} catch {
    throw "TTS service is unavailable at $Server. Start scripts/start-backend.sh and the gateway first. $($_.Exception.Message)"
}

$settings = $Profiles[$Profile].Clone()
if ($null -ne $Temperature) { $settings.Temperature = [double]$Temperature }
if ($null -ne $TopP) { $settings.TopP = [double]$TopP }
if ($null -ne $RepetitionPenalty) { $settings.Repetition = [double]$RepetitionPenalty }
if ($null -ne $MaxTokens) { $settings.MaxTokens = [int]$MaxTokens }
if ($null -ne $Seed) { $settings.Seed = [int]$Seed }
if ($Emotion) { $settings.Emotion = $Emotion }

$voiceId = $References[$Reference]
$outputDirectory = Join-Path $ProjectRoot 'outputs'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$results = @()

for ($index = 0; $index -lt $Lines.Count; $index++) {
    $line = $Lines[$index]
    if ($line.Length -gt 300) {
        throw "Line $($index + 1) has $($line.Length) characters. The current backend limit is 300 characters per request."
    }

    if ($Output) {
        $requestedPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $ProjectRoot $Output }
        if ($Lines.Count -gt 1) {
            $directory = Split-Path -Parent $requestedPath
            $baseName = [IO.Path]::GetFileNameWithoutExtension($requestedPath)
            $requestedPath = Join-Path $directory ('{0}-{1:D3}.wav' -f $baseName, ($index + 1))
        }
        $outputPath = [IO.Path]::ChangeExtension($requestedPath, '.wav')
    } else {
        $outputPath = Join-Path $outputDirectory ('tts-{0}-{1}-{2:D3}.wav' -f $stamp, $Profile.ToLower(), ($index + 1))
    }

    $parent = Split-Path -Parent $outputPath
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }

    $body = @{
        input = $line
        voice = $voiceId
        language = $Language
        emotion = $settings.Emotion
        temperature = $settings.Temperature
        top_p = $settings.TopP
        max_tokens = $settings.MaxTokens
        repetition_penalty = $settings.Repetition
        seed = $settings.Seed
        stream = $false
        response_format = 'wav'
        cache = -not $NoCache
    } | ConvertTo-Json

    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        Invoke-WebRequest `
            -Uri "$($Server.TrimEnd('/'))/v1/audio/speech" `
            -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
            -OutFile $outputPath `
            -TimeoutSec 240 | Out-Null
    } catch {
        $detail = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message } else { $_.Exception.Message }
        throw "Generation failed for line $($index + 1): $detail"
    } finally {
        $timer.Stop()
    }

    $bytes = [IO.File]::ReadAllBytes($outputPath)
    if ($bytes.Length -lt 44 -or [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF') {
        throw "The server did not return a valid WAV for line $($index + 1)."
    }
    $sampleRate = [BitConverter]::ToInt32($bytes, 24)
    $channels = [BitConverter]::ToInt16($bytes, 22)
    $bits = [BitConverter]::ToInt16($bytes, 34)
    $audioSeconds = ($bytes.Length - 44) / ($sampleRate * $channels * ($bits / 8))

    $results += [pscustomobject]@{
        Line = $index + 1
        Profile = $Profile
        Style = $settings.Name
        Reference = $Reference
        Voice = $voiceId
        Temperature = $settings.Temperature
        TopP = $settings.TopP
        Repetition = $settings.Repetition
        Seed = $settings.Seed
        AudioSeconds = [math]::Round($audioSeconds, 2)
        GenerationSeconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
        RTF = [math]::Round($timer.Elapsed.TotalSeconds / $audioSeconds, 2)
        Output = $outputPath
    }
}

$results | Format-List
