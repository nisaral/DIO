$key = "$env:USERPROFILE\.ssh\dio_cloud"
$o = @("-i",$key,"-o","IdentitiesOnly=yes","-o","PreferredAuthentications=publickey","-o","PasswordAuthentication=no","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=20","-o","ServerAliveInterval=30","-o","BatchMode=yes")
$T4 = "216.48.178.114"
$destRoot = "$env:USERPROFILE\OneDrive\Desktop\Go-serve\dio-serve\results_regime_d_cloud"
$log = "$destRoot\backup_pull.log"
New-Item -ItemType Directory -Force -Path "$destRoot\snapshots" | Out-Null
New-Item -ItemType Directory -Force -Path "$destRoot\latest" | Out-Null
function Pull-Once {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $line = "[$stamp] pull start"
  Add-Content $log $line
  try {
    ssh @o root@$T4 "rm -f /tmp/dio_results.tgz; mkdir -p /tmp/dio_pull; rm -rf /tmp/dio_pull/*; cp -a ~/results_regime_d /tmp/dio_pull/ 2>/dev/null; cp -a ~/results_sku_baseline /tmp/dio_pull/ 2>/dev/null; cp -a ~/results_gpu_abc_hetero /tmp/dio_pull/ 2>/dev/null; cp -a ~/regime_d_run.log /tmp/dio_pull/ 2>/dev/null; cp -a ~/extras_after_d.log /tmp/dio_pull/ 2>/dev/null; cd /tmp/dio_pull && tar czf /tmp/dio_results.tgz . ; ls -la /tmp/dio_results.tgz" | Out-String | Add-Content $log
    scp @o "root@${T4}:/tmp/dio_results.tgz" "$destRoot\snapshots\dio_results_$stamp.tgz" 2>&1 | Out-String | Add-Content $log
    scp @o "root@${T4}:~/regime_d_run.log" "$destRoot\regime_d_run.log" 2>&1 | Out-Null
    scp @o "root@${T4}:~/extras_after_d.log" "$destRoot\extras_after_d.log" 2>&1 | Out-Null
    Remove-Item -Recurse -Force "$destRoot\latest\*" -ErrorAction SilentlyContinue
    tar -xzf "$destRoot\snapshots\dio_results_$stamp.tgz" -C "$destRoot\latest" 2>&1 | Out-Null
    $sz = (Get-Item "$destRoot\snapshots\dio_results_$stamp.tgz").Length
    Add-Content $log "[$stamp] OK size=$sz"
    # stop looping after full success markers
    $runlog = Get-Content "$destRoot\regime_d_run.log" -ErrorAction SilentlyContinue -Raw
    $exlog = Get-Content "$destRoot\extras_after_d.log" -ErrorAction SilentlyContinue -Raw
    if ($runlog -match 'EXIT:0' -and (Test-Path "$destRoot\latest\results_regime_d\summary.json")) {
      if ($exlog -match 'ALL EXTRAS DONE' -or $exlog -match 'EXIT') {
        Add-Content $log "[$stamp] FULL DONE - backup loop exiting"
        return $true
      }
      # D done but extras maybe not - keep pulling a while
      if ($runlog -match 'EXIT:0' -and -not (Test-Path "$destRoot\latest\results_sku_baseline")) {
        Add-Content $log "[$stamp] D done, waiting extras"
      }
    }
  } catch {
    Add-Content $log "[$stamp] FAIL $_"
  }
  return $false
}
for ($i=0; $i -lt 80; $i++) {
  $done = Pull-Once
  if ($done) { break }
  Start-Sleep -Seconds 180
}
Add-Content $log "backup loop ended $(Get-Date -Format o)"
