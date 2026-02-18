"""Graceful shutdown callback.

Requests a checkpointed stop when SIGINT/SIGTERM/SIGHUP is received.
"""

import os
import signal
from typing import Dict

import transformers
from transformers.utils import logging

logger = logging.get_logger("transformers")


class GracefulExitCallback(transformers.TrainerCallback):
    """Stop training cleanly after current step when process receives a termination signal."""

    def __init__(self, stop_file_name: str = "STOP_TRAIN"):
        self._stop_requested = False
        self._signal_name = None
        self._orig_handlers: Dict[signal.Signals, object] = {}
        self._handlers_installed = False
        self._log_once = False
        self._stop_file_name = stop_file_name
        self._stop_file_paths = []

    def _request_stop(self, reason: str):
        if self._stop_requested:
            return
        self._stop_requested = True
        self._signal_name = reason
        print(
            f"[GracefulExit] Requested stop ({reason}). "
            "Will stop after current step and save checkpoint.",
            flush=True,
        )

    def _handle_signal(self, signum, frame):
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        self._request_stop(signal_name)

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

    def _check_stop_file(self):
        if self._stop_requested:
            return
        for stop_file in self._stop_file_paths:
            if os.path.exists(stop_file):
                try:
                    os.remove(stop_file)
                except OSError:
                    pass
                self._request_stop(f"STOP_FILE:{stop_file}")
                return

    def on_train_begin(self, args, state, control, **kwargs):
        stop_file_env = os.environ.get("LLAVA_STOP_FILE", "").strip()
        if stop_file_env:
            self._stop_file_paths = [os.path.abspath(stop_file_env)]
        else:
            self._stop_file_paths = [
                os.path.abspath(self._stop_file_name),
                os.path.abspath(os.path.join(args.output_dir, self._stop_file_name)),
            ]
        if state.is_local_process_zero:
            print(
                "[GracefulExit] Stop-file paths: " + ", ".join(self._stop_file_paths),
                flush=True,
            )
        self._install_handlers()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._check_stop_file()
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
