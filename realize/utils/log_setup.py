"""Logging setup for the realize training stack.

Attaches a single stdout handler to the "realize" logger tree, so any
logging.getLogger("realize.<module>").info(...) call writes to the current process
stdout.

- The handler stream is a proxy that resolves sys.stdout on every write/flush, so it
  follows worker processes reassigning sys.stdout to a per-worker file (a normal
  StreamHandler binds its stream once at construction and would not follow).
- INFO records are emitted with NO level prefix. The analysis pipeline parses
  unprefixed, category-tagged INFO lines (e.g. "[PPO] ...", "[STEP] ...") as a data
  interface, so INFO output must stay verbatim and unprefixed.
- WARNING / ERROR / DEBUG get a "[LEVEL] " prefix so they stand out.
"""
import logging
import sys


class _StdoutProxy:
    """File-like object that forwards to the current sys.stdout.

    Binding a handler to this proxy (instead of to sys.stdout directly) lets a
    single handler follow any later reassignment of sys.stdout, as print() does
    by resolving sys.stdout on each call.
    """

    def write(self, data):
        return sys.stdout.write(data)

    def flush(self):
        return sys.stdout.flush()


class _LevelAwareFormatter(logging.Formatter):
    """INFO stays unprefixed and verbatim (the analysis pipeline parses it as a
    data interface); WARNING / ERROR / DEBUG get a "[LEVEL] " prefix.
    """

    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.INFO:
            return message
        return f"[{record.levelname}] {message}"


def configure_logging(verbose=False):
    """Attach a single stdout handler to the "realize" logger.

    verbose=False -> INFO (milestones, summaries, the category-tagged lines the
    analysis pipeline parses); verbose=True -> DEBUG (adds diagnostics).

    Idempotent: repeated calls (re-imported modules in worker processes,
    inherited handler lists under fork) do not add duplicate handlers, but the
    level IS refreshed each call so a later call can switch normal<->verbose.
    """
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger("realize")
    root.setLevel(level)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and isinstance(
            getattr(handler, "stream", None), _StdoutProxy
        ):
            # Handler already present (e.g. configured before config load).
            # Refresh its level so a later verbose/normal decision takes effect.
            handler.setLevel(level)
            return
    handler = logging.StreamHandler(_StdoutProxy())
    handler.setFormatter(_LevelAwareFormatter("%(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
