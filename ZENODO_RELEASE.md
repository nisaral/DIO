# Zenodo and GitHub release procedure

The repository contains `.zenodo.json`, `CITATION.cff`, and
`dio-serve/scripts/build_artifact_release.ps1`. The release script excludes local
virtual environments and vendored smoke-test checkouts.

## Local validation

```powershell
& .\dio-serve\scripts\build_artifact_release.ps1 -Version 0.3.0-rc1 -Out artifact_release
Get-FileHash .\artifact_release\dio-0.3.0-rc1.zip -Algorithm SHA256
```

## Publish on GitHub

After authenticating `gh` or configuring a remote credential:

```powershell
git add .zenodo.json CITATION.cff ZENODO_RELEASE.md RELEASE_NOTES.md dio-serve paper_drafts_latex
git commit -m "Prepare calibration-robust DIO journal revision"
git tag -a v0.3.0-rc1 -m "DIO artifact release 0.3.0-rc1"
git push origin HEAD --tags
gh release create v0.3.0-rc1 .\artifact_release\dio-0.3.0-rc1.zip --title "DIO 0.3.0-rc1" --notes-file RELEASE_NOTES.md
```

## Publish on Zenodo

1. Enable the GitHub repository in Zenodo and select the `v0.3.0-rc1` release.
2. Confirm metadata imported from `.zenodo.json` and attach the release archive.
3. Publish, copy the assigned DOI into the manuscript and citation files, and make
   a small follow-up release if required.

The current environment has no valid GitHub or Zenodo credential, so no external
push or DOI minting is performed automatically.
