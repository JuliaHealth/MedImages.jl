#!/bin/bash
rm -f mem.log
while true; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> mem.log
  sleep 0.1
done
