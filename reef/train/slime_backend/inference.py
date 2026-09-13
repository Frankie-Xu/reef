"""Translate legacy Slime inference flags into plain deployment input values."""

from __future__ import annotations

from typing import Any

from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars


def inference_config(args: Any) -> dict[str, Any]:
    """Return launch input without importing or constructing an inference backend."""
    options = {
        key.removeprefix("sglang_"): value
        for key, value in vars(args).items()
        if key.startswith("sglang_")
        and not key.startswith("sglang_router_")
        and key not in {"sglang_config", "sglang_model_routers"}
    }
    options.update(
        model_path=args.hf_checkpoint,
        trust_remote_code=True,
        random_seed=getattr(args, "seed", 1),
        enable_memory_saver=bool(getattr(args, "offload_rollout", False)),
        enable_draft_weights_cpu_backup=True,
        skip_server_warmup=True,
        enable_metrics=True,
    )
    if getattr(args, "fp16", False):
        options["dtype"] = "float16"
    if getattr(args, "use_rollout_routing_replay", False):
        options["enable_return_routed_experts"] = True
    rank = int(getattr(args, "megatron_lora_rank", 0) or 0)
    if rank > 0:
        from reef.train.slime_backend.reef_adapters.megatron.lora import sglang_lora_target_modules

        configured_slots = getattr(args, "max_loaded_loras", None)
        slots = 1 if configured_slots is None else int(configured_slots)
        if slots < 1:
            raise ValueError("--max-loaded-loras must be at least 1")
        options.update(
            enable_lora=True,
            max_lora_rank=rank,
            max_loaded_loras=slots,
            max_loras_per_batch=slots,
            lora_target_modules=sglang_lora_target_modules(args),
            enable_weights_cpu_backup=True,
            tokenizer_worker_num=1,
        )
    models: tuple[dict[str, Any], ...] = ()
    config_path = getattr(args, "sglang_config", None)
    if config_path:
        # Slime's CLI still accepts its legacy group file. Resolve it only at
        # this input boundary; inference receives ordinary Reef values.
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        parsed = SglangConfig.from_yaml(config_path)
        for model in parsed.models:
            model.resolve(args)
        models = tuple(
            {
                "name": model.name,
                "groups": tuple(
                    {
                        "worker_type": group.worker_type,
                        "num_gpus": group.num_gpus,
                        "gpus_per_engine": group.num_gpus_per_engine,
                        "options": {key.replace("-", "_"): value for key, value in group.overrides.items()},
                    }
                    for group in model.server_groups
                ),
                "update_weights": bool(model.update_weights),
            }
            for model in parsed.models
        )
    elif getattr(args, "prefill_num_servers", 0):
        prefill = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        models = (
            {
                "name": "default",
                "groups": (
                    {
                        "worker_type": "prefill",
                        "num_gpus": prefill,
                        "gpus_per_engine": args.rollout_num_gpus_per_engine,
                    },
                    {
                        "worker_type": "decode",
                        "num_gpus": args.rollout_num_gpus - prefill,
                        "gpus_per_engine": args.rollout_num_gpus_per_engine,
                    },
                ),
            },
        )
    env = reef_rollout_env_vars()
    if "SLIME_HOST_IP" in env:
        env["REEF_INFERENCE_HOST"] = env.pop("SLIME_HOST_IP")
    return {
        "model_path": args.hf_checkpoint,
        "num_gpus": args.rollout_num_gpus,
        "gpus_per_engine": args.rollout_num_gpus_per_engine,
        "gpus_per_node": args.num_gpus_per_node,
        "options": options,
        "models": models,
        "external_engines": tuple(getattr(args, "rollout_external_engine_infos", ()) or ()),
        "router_host": getattr(args, "sglang_router_ip", None),
        "router_port": getattr(args, "sglang_router_port", None),
        "router_options": {
            key.removeprefix("sglang_router_"): value
            for key, value in vars(args).items()
            if key.startswith("sglang_router_") and key not in {"sglang_router_ip", "sglang_router_port"}
        },
        "env_vars": env,
        "offload": bool(getattr(args, "offload_rollout", False)),
        "shared_gpus": args.actor_num_nodes * args.actor_num_gpus_per_node if getattr(args, "colocate", False) else 0,
        "check_weights": bool(getattr(args, "check_weight_update_equal", False)),
        "pause_mode": "retract" if getattr(args, "colocate", False) else "in_place",
        "health_enabled": bool(getattr(args, "use_fault_tolerance", False)),
        "health_interval": getattr(args, "rollout_health_check_interval", 30),
        "health_timeout": getattr(args, "rollout_health_check_timeout", 30),
        "health_first_wait": getattr(args, "rollout_health_check_first_wait", 60),
        "request_timeout": getattr(args, "distributed_timeout_minutes", 10) * 60,
        "executor": getattr(args, "reef_rollout_executor_backend", "auto"),
        "executor_options": getattr(args, "reef_rollout_executor_options", {}),
    }
