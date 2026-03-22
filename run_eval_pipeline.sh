#!/bin/bash
echo "Waiting for train_ude.jl (PID 241501) to finish..."
while kill -0 241501 2>/dev/null; do
    sleep 60
done
echo "Training finished. Starting evaluation..."
/home/user/miniforge/envs/sciml_env/bin/julia --project=. elsarticle/dosimetry/eval_model.jl > elsarticle/dosimetry/eval_output.log 2>&1
echo "Evaluation finished. Starting visualization plotting..."
python elsarticle/dosimetry/plot_results.py > elsarticle/dosimetry/plot_output.log 2>&1
echo "Pipeline completed successfully!"
