export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export XLA_FLAGS="--xla_gpu_autotune_level=0" && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/conrft \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=/home/dx/waylen/conrft/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2026-02-05_13-55-18.pkl \
    --pretrain_steps=2000 \
    --debug=False \
    --learner \
