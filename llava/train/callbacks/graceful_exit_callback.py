"""Graceful shutdown callback.

Requests a checkpointed stop when SIGINT/SIGTERM/SIGHUP is received.
"""

import signal
from typing import Dict

import transformers
from transformers.utils import logging

logger = logging.get_logger("transformers")


class GracefulExitCallback(transformers.TrainerCallback):
    """Stop training cleanly after current step when process receives a termination signal."""

    def __init__(self):
        self._stop_requested = False
        self._signal_name = None
        self._orig_handlers: Dict[signal.Signals, object] = {}
        self._handlers_installed = False
        self._log_once = False

    def _handle_signal(self, signum, frame):
        if self._stop_requested:
            return
        self._stop_requested = True
        try:
            self._signal_name = signal.Signals(signum).name
        except Exception:
            self._signal_name = str(signum)
        print(
            f"[GracefulExit] Received {self._signal_name}. "
            "Will stop after current step and save checkpoint.",
            flush=True,
        )

    def _install_handlers(self):
        if self._handlers_installed:
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            signals.append(signal.SIGHUP)
        for sig in signals:
            self._orig_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        self._handlers_installed = True

    def _restore_handlers(self):
        if not self._handlers_installed:
            return
        for sig, handler in self._orig_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        self._handlers_installed = False

    def on_train_begin(self, args, state, control, **kwargs):
        self._install_handlers()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._stop_requested:
            control.should_save = True
            control.should_training_stop = True
            if not self._log_once and state.is_local_process_zero:
                logger.warning(
                    f"Graceful exit requested by {self._signal_name}; "
                    f"saving checkpoint at global_step={state.global_step}."
                )
                self._log_once = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._restore_handlers()
        return control

