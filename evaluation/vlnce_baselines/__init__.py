from vlnce_baselines import cma_trainer, dagger_trainer, ddppo_waypoint_trainer, recollect_trainer
from vlnce_baselines import qwen3vl_gru_trainer  # Qwen3-VL + spatial-memory GRU (no llava needed)
from vlnce_baselines.common import environments
from vlnce_baselines.models import cma_policy, seq2seq_policy, waypoint_policy

try:  # VILA/NaVILA's own trainer; only importable in an env with llava installed
    from vlnce_baselines import navila_trainer
except ModuleNotFoundError:
    navila_trainer = None
