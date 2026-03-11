$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "primordial_venv\Scripts\python.exe"
$distRoot = Join-Path $root "dist"
$buildRoot = Join-Path $root "build"
$appName = "PrimordialObservatory"
$appDist = Join-Path $distRoot $appName
$stageRoot = Join-Path $buildRoot "setup_stage"
$stageApp = Join-Path $stageRoot "app"
$setupExe = Join-Path $distRoot "Primordial_Observatory_Setup.exe"
$sedPath = Join-Path $buildRoot "PrimordialObservatorySetup.sed"
$issPath = Join-Path $buildRoot "PrimordialObservatorySetup.iss"
$installCmd = Join-Path $stageRoot "install_observatory.cmd"
$innoCandidates = @(
    "C:\Users\pc\AppData\Local\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)
$iscc = $innoCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

function Get-RelativePathCompat {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $base = [IO.Path]::GetFullPath($BasePath)
    if (-not $base.EndsWith("\")) {
        $base += "\"
    }
    $target = [IO.Path]::GetFullPath($TargetPath)
    return $target.Substring($base.Length)
}

if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}

Write-Host "[1/5] Ensuring PyInstaller is installed..."
& $python -m pip install pyinstaller | Out-Host

Write-Host "[2/5] Cleaning previous build artifacts..."
if (Test-Path $appDist) { Remove-Item $appDist -Recurse -Force }
if (Test-Path $stageRoot) { Remove-Item $stageRoot -Recurse -Force }
if (Test-Path $setupExe) { Remove-Item $setupExe -Force }
New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
New-Item -ItemType Directory -Force -Path $stageApp | Out-Null

Write-Host "[3/5] Building standalone executable..."
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $appName `
    --paths $root `
    --collect-all taichi `
    --collect-all dearpygui `
    --collect-submodules primordial `
    --add-data "$root\primordial_ppo_v17.pt;." `
    "$root\primordial\apps\observatory\app.py" | Out-Host

$exePath = Join-Path $appDist "$appName.exe"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

for ($i = 0; $i -lt 10 -and -not (Test-Path $exePath); $i++) {
    Start-Sleep -Milliseconds 500
}

if (-not (Test-Path $exePath)) {
    throw "PyInstaller did not produce $appName.exe"
}

Write-Host "[4/5] Staging installer payload..."
robocopy $appDist $stageApp /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed while staging app payload"
}

$installScript = @'
@echo off
setlocal
set "TARGET=%LOCALAPPDATA%\Programs\PrimordialObservatory"
if not exist "%TARGET%" mkdir "%TARGET%"
robocopy "%~dp0app" "%TARGET%" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$desk=[Environment]::GetFolderPath('Desktop'); $target=Join-Path $env:LOCALAPPDATA 'Programs\PrimordialObservatory\PrimordialObservatory.exe'; $shell=New-Object -ComObject WScript.Shell; $shortcut=$shell.CreateShortcut((Join-Path $desk 'Primordial Observatory.lnk')); $shortcut.TargetPath=$target; $shortcut.WorkingDirectory=[IO.Path]::GetDirectoryName($target); $shortcut.IconLocation=$target; $shortcut.Save()"
start "" "%TARGET%\PrimordialObservatory.exe"
exit /b 0
'@
Set-Content -Path $installCmd -Value $installScript -Encoding ASCII

$allFiles = Get-ChildItem -Path $stageRoot -Recurse -File | Sort-Object FullName
$groups = @{}
$index = 0

foreach ($file in $allFiles) {
    $parent = $file.DirectoryName
    if (-not $groups.ContainsKey($parent)) {
        $groups[$parent] = @()
    }
    $groups[$parent] += [PSCustomObject]@{
        Id       = $index
        Relative = Get-RelativePathCompat -BasePath $stageRoot -TargetPath $file.FullName
    }
    $index++
}

$strings = @()
foreach ($group in $groups.GetEnumerator() | Sort-Object Name) {
    foreach ($entry in $group.Value) {
        $strings += "FILE$($entry.Id)=$($entry.Relative)"
    }
}

$sed = @(
    "[Version]",
    "Class=IEXPRESS",
    "SEDVersion=3",
    "",
    "[Options]",
    "PackagePurpose=InstallApp",
    "ShowInstallProgramWindow=0",
    "HideExtractAnimation=1",
    "UseLongFileName=1",
    "InsideCompressed=0",
    "CAB_FixedSize=0",
    "CAB_ResvCodeSigning=0",
    "RebootMode=N",
    "InstallPrompt=",
    "DisplayLicense=",
    "FinishMessage=Primordial Observatory installed.",
    "TargetName=$setupExe",
    "FriendlyName=Primordial Observatory Setup",
    "AppLaunched=install_observatory.cmd",
    "PostInstallCmd=<None>",
    "AdminQuietInstCmd=",
    "UserQuietInstCmd=",
    "SourceFiles=SourceFiles",
    "",
    "[Strings]"
)

$sed += $strings
$sed += @("", "[SourceFiles]")

$sectionIndex = 0
foreach ($group in $groups.GetEnumerator() | Sort-Object Name) {
    $sed += "SourceFiles$sectionIndex=$($group.Key)"
    $sectionIndex++
}

$sectionIndex = 0
foreach ($group in $groups.GetEnumerator() | Sort-Object Name) {
    $sed += @("", "[SourceFiles$sectionIndex]")
    foreach ($entry in $group.Value) {
        $sed += "%FILE$($entry.Id)%="
    }
    $sectionIndex++
}

Set-Content -Path $sedPath -Value $sed -Encoding ASCII

Write-Host "[5/5] Building Setup.exe..."
if ($iscc) {
    $iss = @"
#define MyAppName "Primordial Observatory"
#define MyAppVersion "17.2"
#define MyAppPublisher "PRIMORDIAL CORE"
#define MyAppExeName "PrimordialObservatory.exe"

[Setup]
AppId={{5BDF6557-2C98-4E96-89B0-2E0D37E2C5A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\PrimordialObservatory
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=$distRoot
OutputBaseFilename=Primordial_Observatory_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "$appDist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\Primordial Observatory"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\Primordial Observatory"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Primordial Observatory"; Flags: nowait postinstall skipifsilent
"@
    Set-Content -Path $issPath -Value $iss -Encoding ASCII
    & $iscc $issPath | Out-Host
} else {
    & iexpress /N $sedPath | Out-Host
}

if (-not (Test-Path $setupExe)) {
    throw "IExpress did not produce setup executable"
}

Write-Host ""
Write-Host "Build complete:"
Write-Host "  EXE   : $appDist\$appName.exe"
Write-Host "  SETUP : $setupExe"
