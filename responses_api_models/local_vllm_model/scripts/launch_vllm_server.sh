

# ----- PARAMETERS -----
# EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION, CONTAINER_IMAGE_PATH, SLURM_ACCOUNT, SLURM_PARTITION

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION

# Construct the command
read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

hostname -i

cd 3rdparty/Gym-workspace/Gym
source .venv/bin/activate

ng_run "+config_paths=[responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml]" \
    ++use_absolute_ip=true

EOF

echo -e "Running command:\n$COMMAND"

mount=$(findmnt -n -o TARGET --target .)

COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS=$mount:$mount \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
