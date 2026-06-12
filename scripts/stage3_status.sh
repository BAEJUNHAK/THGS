#!/usr/bin/env bash
# Stage 3.1 실험 실시간 상태판.  사용:  watch -n 10 bash scripts/stage3_status.sh
cd "$(dirname "$0")/.." || exit 1

echo "════════ Stage 3.1 — B8 인과 replay 실험 상태 ($(date +%H:%M:%S)) ════════"
echo
echo "── Phase A: language_features 인코딩 ──"
total_remain=0
for sc in ramen figurines teatime waldo_kitchen; do
    n=$(ls data/lerf_ovs/$sc/language_features 2>/dev/null | wc -l)
    imgs=$(ls data/lerf_ovs/$sc/images | wc -l)
    t=$((imgs * 2))
    pct=$((n * 100 / t))
    done_imgs=$((n / 2))
    remain=$((imgs - done_imgs))
    total_remain=$((total_remain + remain))
    bar=$(printf '#%.0s' $(seq 1 $((pct / 5)) 2>/dev/null))
    if [ "$n" -ge "$t" ]; then
        printf "  %-14s [%-20s] %3d%%  DONE\n" "$sc" "####################" "$pct"
    else
        rate=$(tail -c 300 output/diagnostics/logs/encode_$sc.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+\.[0-9]+s/it' | tail -1)
        printf "  %-14s [%-20s] %3d%%  %d/%d imgs  %s\n" "$sc" "$bar" "$pct" "$done_imgs" "$imgs" "${rate:-}"
    fi
done
echo
echo "── GPU 2 ──"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed -n 3p | awk -F, '{printf "  util:%s  mem:%s /%s\n", $1, $2, $3}'
echo
echo "── Phase B: replay 충실도 (cos≥0.95 gate) ──"
for log in output/diagnostics/logs/stage3_replay_*.log; do
    [ -f "$log" ] || continue
    sc=$(basename "$log" .log | sed 's/stage3_replay_//')
    pass=$(grep -c "PASS" "$log" 2>/dev/null)
    fail=$(grep -c "FAIL" "$log" 2>/dev/null)
    echo "  $sc: PASS=$pass FAIL=$fail"
done
echo
echo "── 최근 활동 ──"
for sc in figurines teatime waldo_kitchen; do
    line=$(tail -c 200 output/diagnostics/logs/encode_$sc.log 2>/dev/null | tr '\r' '\n' | grep "it \[" | tail -1)
    [ -n "$line" ] && echo "  $sc: $line"
done
echo
echo "다음 단계: 인코딩 4/4 완료 → scene별 replay(~12s) → 17 phantom 최종 판정 (H-B8a/b/c)"
