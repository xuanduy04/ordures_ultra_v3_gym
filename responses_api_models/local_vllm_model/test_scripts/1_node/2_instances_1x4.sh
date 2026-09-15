

config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml,\
responses_api_models/local_vllm_model/configs/openai/gpt-oss-120b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4 \
    ++gpt-oss-120b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4
