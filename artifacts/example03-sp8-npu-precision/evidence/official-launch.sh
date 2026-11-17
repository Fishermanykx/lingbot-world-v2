#!/bin/bash
set -eo pipefail
R=$(cd "$(dirname "$0")" && pwd)
cd "$R/source"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/deps:/mnt/share/workspace/experiments/lingbot-mxfp8-layer-rollback-22-20261117/deps${PYTHONPATH:+:$PYTHONPATH}"
prompt=$(sed -n 's/.*--prompt "\(.*\)" .*/\1/p' run_fast.sh)
test -n "$prompt"
exec torchrun --nproc_per_node=8 --master_port=29861 generate.py --task i2v-A14B --size '480*832' \
  --ckpt_dir lingbot-world-v2-14b-causal-fast --image examples/03/image.jpg --action_path examples/03 \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 361 --base_seed 42 --local_attn_size 18 --sink_size 6 \
  --prompt "$prompt" --save_file "$R/official-example03.mp4"
