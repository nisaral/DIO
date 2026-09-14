#Requires -Version 5.1
<#
.SYNOPSIS
    Applies the launch metadata for the DIO GitHub repository.

.DESCRIPTION
    Sets the repo description, homepage (Zenodo DOI), topics, and enables Issues
    and Discussions. Idempotent: it reads the current state first and only sends
    changes for the fields that actually differ, so it is safe to re-run.

    It uses the locally authenticated `gh` CLI. No token is read, stored, or
    printed by this script.

.PARAMETER Repo
    Target repository in owner/name form. Defaults to nisaral/DIO.

.PARAMETER DryRun
    Print the commands that would run, without calling the GitHub API.

.PARAMETER SkipDiscussions
    Do not touch the Discussions setting. Useful if Discussions is unavailable
    for the account or org.

.EXAMPLE
    .\gh_repo_setup.ps1 -DryRun
    Shows what would change without changing anything.

.EXAMPLE
    .\gh_repo_setup.ps1
    Applies the metadata.

.NOTES
    Companion document: docs/launch/REPO_METADATA.md
    Script is intentionally ASCII-only for cross-platform safety.
#>

[CmdletBinding()]
param(
    [string]$Repo = 'nisaral/DIO',
    [switch]$DryRun,
    [switch]$SkipDiscussions
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Values to apply. Keep in sync with docs/launch/REPO_METADATA.md.
# GitHub hard-limits the description to 160 characters and topics to 20.
# ---------------------------------------------------------------------------

$Description = 'DIO: non-invasive LLM gateway that learns per-backend latency online (dual-timescale NLMS) and routes vLLM/Ollama/SGLang/TGI with SLO-aware admission.'

$Homepage = 'https://doi.org/10.5281/zenodo.22085398'

$Topics = @(
    'llm'
    'llm-serving'
    'llm-inference'
    'llm-gateway'
    'llmops'
    'mlops'
    'vllm'
    'ollama'
    'sglang'
    'openai-api'
    'openai-compatible'
    'model-serving'
    'inference-server'
    'load-balancing'
    'admission-control'
    'model-context-protocol'
    'mcp-server'
    'ai-infrastructure'
    'self-hosted'
    'python'
)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Info {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "[dio] $Message"
}

function Write-Change {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "[dio] CHANGE  $Message"
}

function Write-Keep {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "[dio] ok      $Message"
}

function Invoke-Gh {
    param(
        [Parameter(Mandatory)][string[]]$GhArgs,
        [Parameter(Mandatory)][string]$Label
    )
    $pretty = 'gh ' + ($GhArgs -join ' ')
    if ($DryRun) {
        Write-Change "$Label : DRY RUN, would execute: $pretty"
        return
    }
    Write-Change "$Label"
    Write-Host "        $pretty"
    & gh @GhArgs
    if ($LASTEXITCODE -ne 0) {
        throw "gh failed with exit code $LASTEXITCODE for: $pretty"
    }
}
function Get-Prop {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

Write-Info "DIO launch metadata for '$Repo'"
if ($DryRun) { Write-Info 'DRY RUN enabled: no changes will be sent to GitHub.' }

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'gh CLI not found on PATH. Install it from https://cli.github.com/ and run "gh auth login".'
}

if ($Description.Length -gt 160) {
    throw "Description is $($Description.Length) characters; GitHub allows 160."
}
if ($Topics.Count -gt 20) {
    throw "Topic list has $($Topics.Count) entries; GitHub allows 20."
}

& gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'gh is not authenticated. Run "gh auth login" first. This script never handles tokens itself.'
}
Write-Keep "gh present and authenticated; description is $($Description.Length)/160 chars; $($Topics.Count)/20 topics"

# ---------------------------------------------------------------------------
# Read current state (idempotency depends on this)
# ---------------------------------------------------------------------------

$jsonFields = 'description,homepageUrl,repositoryTopics,hasIssuesEnabled,hasDiscussionsEnabled'
$raw = & gh repo view $Repo --json $jsonFields 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Could not read '$Repo'. Check the owner/name and your access."
}
$current = $raw | ConvertFrom-Json

$currentDescription = Get-Prop $current 'description'
$currentHomepage = Get-Prop $current 'homepageUrl'

$currentTopics = @()
$topicObjects = Get-Prop $current 'repositoryTopics'
if ($topicObjects) {
    $currentTopics = @($topicObjects | ForEach-Object { $_.name })
}

Write-Info "Current topics ($($currentTopics.Count)): $($currentTopics -join ', ')"

# ---------------------------------------------------------------------------
# 1. Description
# ---------------------------------------------------------------------------

if ($currentDescription -eq $Description) {
    Write-Keep 'Description already correct'
}
else {
    Invoke-Gh -Label 'Set description' -GhArgs @('repo', 'edit', $Repo, '--description', $Description)
}

# ---------------------------------------------------------------------------
# 2. Homepage (Zenodo DOI)
# ---------------------------------------------------------------------------

if ($currentHomepage -eq $Homepage) {
    Write-Keep "Homepage already $Homepage"
}
else {
    Invoke-Gh -Label 'Set homepage' -GhArgs @('repo', 'edit', $Repo, '--homepage', $Homepage)
}

# ---------------------------------------------------------------------------
# 3. Topics
# ---------------------------------------------------------------------------

# $Topics is the *desired final set*, not an additive wish list: otherwise a repo
# that already carries loosely-related topics can never converge on the curated
# 20 (GitHub hard-caps topics at 20), and re-running the script just fails.
if ($Topics.Count -gt 20) {
    throw "The topic list has $($Topics.Count) entries; GitHub allows at most 20."
}

$missingTopics = @($Topics | Where-Object { $currentTopics -notcontains $_ })
$extraTopics = @($currentTopics | Where-Object { $Topics -notcontains $_ })

if ($missingTopics.Count -eq 0 -and $extraTopics.Count -eq 0) {
    Write-Keep "All $($Topics.Count) topics already applied"
}
else {
    $topicArgs = @('repo', 'edit', $Repo)
    foreach ($topic in $missingTopics) {
        $topicArgs += @('--add-topic', $topic)
    }
    foreach ($topic in $extraTopics) {
        $topicArgs += @('--remove-topic', $topic)
    }

    $label = "Topics: add $(@($missingTopics).Count), remove $(@($extraTopics).Count)"
    if ($missingTopics.Count -gt 0) {
        $label += " (add: $($missingTopics -join ', '))"
    }
    if ($extraTopics.Count -gt 0) {
        $label += " (remove: $($extraTopics -join ', '))"
    }
    Invoke-Gh -Label $label -GhArgs $topicArgs
}
# ---------------------------------------------------------------------------
# 4. Issues and Discussions
#
# `gh repo edit` exposes --enable-issues/--enable-wiki/--enable-projects but has
# no Discussions flag, so Discussions is toggled through the REST API
# (has_discussions on the repository object). Both calls are PATCH and are
# no-ops when the value is already what we want.
# ---------------------------------------------------------------------------

$needIssues = -not (Get-Prop $current 'hasIssuesEnabled')

$needDiscussions = $false
if (-not $SkipDiscussions) {
    $discussionsValue = Get-Prop $current 'hasDiscussionsEnabled'
    if ($null -eq $discussionsValue) {
        Write-Info 'Could not read hasDiscussionsEnabled from this gh version; attempting to enable it anyway.'
        $needDiscussions = $true
    }
    elseif (-not $discussionsValue) {
        $needDiscussions = $true
    }
}

if (-not $needIssues -and -not $needDiscussions) {
    Write-Keep 'Issues and Discussions already enabled'
}
else {
    $apiArgs = @('api', '--method', 'PATCH', "repos/$Repo")
    $labels = @()
    if ($needIssues) {
        $apiArgs += @('-F', 'has_issues=true')
        $labels += 'issues'
    }
    if ($needDiscussions) {
        $apiArgs += @('-F', 'has_discussions=true')
        $labels += 'discussions'
    }
    Invoke-Gh -Label "Enable $($labels -join ' and ')" -GhArgs $apiArgs
}

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------

if ($DryRun) {
    Write-Info 'Dry run complete. Re-run without -DryRun to apply.'
    exit 0
}

$rawAfter = & gh repo view $Repo --json $jsonFields
$after = $rawAfter | ConvertFrom-Json
$afterTopics = @()
$afterTopicObjects = Get-Prop $after 'repositoryTopics'
if ($afterTopicObjects) { $afterTopics = @($afterTopicObjects | ForEach-Object { $_.name }) }

Write-Host ''
Write-Info 'Result'
Write-Host "  description : $(Get-Prop $after 'description')"
Write-Host "  homepage    : $(Get-Prop $after 'homepageUrl')"
Write-Host "  topics      : $($afterTopics.Count) -> $($afterTopics -join ', ')"
Write-Host "  issues      : $(Get-Prop $after 'hasIssuesEnabled')"
Write-Host "  discussions : $(Get-Prop $after 'hasDiscussionsEnabled')"

$stillMissing = @($Topics | Where-Object { $afterTopics -notcontains $_ })
if ($stillMissing.Count -gt 0) {
    Write-Warning "Topics still missing after the run: $($stillMissing -join ', ')"
}

# ---------------------------------------------------------------------------
# 6. Manual steps this script cannot do
# ---------------------------------------------------------------------------

Write-Host ''
Write-Info 'Still manual (see docs/launch/REPO_METADATA.md):'
Write-Host '  1. Upload the social preview image (Settings -> General -> Social preview).'
Write-Host '  2. Pin the repo on your profile.'
Write-Host '  3. Add the demo GIF and the "why not Nginx round-robin?" paragraph to the READMEs.'
Write-Host '  4. Cut the v0.4.0 pre-release and paste the RELEASE_NOTES body.'
Write-Host '  5. Start a "Show your setup" Discussion so the tab is not empty.'
