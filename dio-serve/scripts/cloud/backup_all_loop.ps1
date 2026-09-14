$key = "$env:USERPROFILE\.ssh\dio_cloud"
$o = @("-i",$key,"-o","IdentitiesOnly=yes","-o","PreferredAuthentications=publickey","-o","PasswordAuthentication=no","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=25","-o","BatchMode=yes")
$T4="216.48.178.114"
$dest="$env:USERPROFILE\OneDrive\Desktop\Go-serve\dio-serve\results_regime_d_cloud"
New-Item -ItemType Directory -Force -Path "$dest\snapshots","$dest\latest" | Out-Null
$log="$dest\backup_pull.log"
for ($i=0; $i -lt 100; $i++) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  try {
    ssh @o root@$T4 "rm -f /tmp/dio_all.tgz; tar czf /tmp/dio_all.tgz -C /root --exclude=dio-venv --exclude=dio-gw-venv results_regime_d results_gpu_abc_hetero results_post_d_priority results_sku_baseline results_sharegpt_hetero results_sharegpt_long results_sharegpt_n3 results_n3_hetero results_extra_seeds_200 results_discovery_strict results_long_decode_d1 results_gpu_abc_hetero_mt96 ALL_DIO_RESULTS.tgz regime_d_run.log priority_followups.log extended_followups.log autonomous_soak.log 2>/dev/null; ls -la /tmp/dio_all.tgz" | Out-String | ForEach-Object { Add-Content $log "[$stamp] $_" }
    scp @o "root@${T4}:/tmp/dio_all.tgz" "$dest\snapshots\all_$stamp.tgz" 2>$null
    scp @o "root@${T4}:~/priority_followups.log" "$dest\priority_followups.log" 2>$null
    scp @o "root@${T4}:~/extended_followups.log" "$dest\extended_followups.log" 2>$null
    scp @o "root@${T4}:~/autonomous_soak.log" "$dest\autonomous_soak.log" 2>$null
    scp @o "root@${T4}:~/regime_d_run.log" "$dest\regime_d_run.log" 2>$null
    Add-Content $log "[$stamp] OK"
    $al = Get-Content "$dest\autonomous_soak.log" -Raw -EA SilentlyContinue
    if ($al -match 'AUTONOMOUS SOAK COMPLETE') { Add-Content $log "soak complete exit"; break }
  } catch { Add-Content $log "[$stamp] FAIL $_" }
  Start-Sleep -Seconds 180
}
