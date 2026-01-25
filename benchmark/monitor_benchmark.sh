#!/bin/bash
# Monitor CPU+GPU Benchmark Progress

LOG_FILE="/workspaces/MedImages.jl/benchmark/cpu_gpu_comparison.log"
STATUS_FILE="/workspaces/MedImages.jl/benchmark/benchmark_status.txt"

echo "=== Benchmark Monitoring Started ===" > "$STATUS_FILE"
date >> "$STATUS_FILE"
echo "" >> "$STATUS_FILE"

# Monitor for 30 minutes, checking every 30 seconds
for i in {1..60}; do
    sleep 30
    
    if [ -f "$LOG_FILE" ]; then
        LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
        SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
        
        echo "--- Check #$i at $(date) ---" >> "$STATUS_FILE"
        echo "Log lines: $LINES, Size: $SIZE" >> "$STATUS_FILE"
        
        # Check if benchmark is running
        if ps aux | grep -q "[j]ulia.*run_gpu_benchmarks"; then
            echo "Status: Running" >> "$STATUS_FILE"
        else
            echo "Status: Process not found" >> "$STATUS_FILE"
        fi
        
        # Show last few lines
        echo "Recent output:" >> "$STATUS_FILE"
        tail -5 "$LOG_FILE" 2>/dev/null >> "$STATUS_FILE"
        echo "" >> "$STATUS_FILE"
    else
        echo "Check #$i: Log file not yet created" >> "$STATUS_FILE"
    fi
done

echo "=== Monitoring Complete ===" >> "$STATUS_FILE"
date >> "$STATUS_FILE"
