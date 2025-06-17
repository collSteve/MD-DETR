import hydra, shlex, subprocess, pathlib, os
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="defaults")
def main(cfg: DictConfig):
    # -------- derive paths --------
    out_dir = pathlib.Path(cfg.shared.base_run_dir) / cfg.experiment.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sbatch gpu per node: {cfg.sbatch.gpus_per_node}")
    print(f"sbatch nodes: {cfg.sbatch.nodes}")

    # -------- build CLI --------
    common = [
        "--output_dir",    str(out_dir),
        "--train_img_dir", cfg.shared.train_dir,
        "--test_img_dir",  cfg.shared.val_dir,
        "--task_ann_dir",  f"{cfg.shared.task_ann_root}/{cfg.experiment.split_point}",
        "--repo_name",     cfg.experiment.repo_name,
        "--n_gpus",        str(cfg.sbatch.gpus_per_node * cfg.sbatch.nodes),
        "--batch_size",    str(cfg.experiment.batch_size),
        "--lr",            str(cfg.experiment.lr),
        "--lr_old",        str(cfg.experiment.lr_old),
        "--n_classes",     "81",
        "--num_workers",   "2",
        "--split_point",   str(cfg.experiment.split_point),
        "--use_prompts",   str(int(cfg.experiment.use_prompts)),
        "--num_prompts",   str(cfg.experiment.num_prompts),
        "--prompt_len",    str(cfg.experiment.plen),
        "--freeze",        cfg.experiment.freeze,
        "--new_params",    cfg.experiment.new_params,
        "--start_task",    str(cfg.experiment.start_task),
        "--n_tasks",       str(cfg.experiment.n_tasks),
        "--local_query",    str(cfg.experiment.local_query),
        "--checkpoint_dir", cfg.experiment.checkpoint_dir,
        "--checkpoint_base",cfg.experiment.checkpoint_base,
        "--checkpoint_next",cfg.experiment.checkpoint_next,

    ]

    if cfg.experiment.viz:
        common.append(cfg.experiment.viz)

    if cfg.experiment.train:
        mode = [
            "--epochs",         str(cfg.experiment.epochs),
            "--save_epochs",    str(cfg.experiment.save_epochs),
            "--eval_epochs",    str(cfg.experiment.eval_epochs),
            "--bg_thres",       str(cfg.experiment.bg_thres),
            "--bg_thres_topk",  str(cfg.experiment.bg_thres_topk),
            "--lambda_query",   str(cfg.experiment.lambda_query),
            "--resume",         str(cfg.experiment.resume),
        ]
    else:
        mode = [
            "--eval",
            "--epochs",         str(cfg.experiment.epochs),
            "--save_epochs",    str(cfg.experiment.save_epochs),
            "--eval_epochs",    str(cfg.experiment.eval_epochs),
            
        ]

    torchrun = [
        "python",
        # "--nnodes",          str(cfg.sbatch.nodes),
        # "--nproc_per_node",  str(cfg.sbatch.gpus_per_node),
        "main.py",
    ] + common + mode

    is_submitit_worker = os.environ.get("SUBMITIT_EXECUTOR") == "slurm"

    if is_submitit_worker or cfg.run.local:
        # we are on the compute node → launch the experiment
        print(" ".join(shlex.quote(x) for x in torchrun))
        subprocess.check_call(torchrun)
    else:
        # we are on the login / driver process
        print("[submitit] Job script generated and submitted to Slurm.")

if __name__ == "__main__":
    main()