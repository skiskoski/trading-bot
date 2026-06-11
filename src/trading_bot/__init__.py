"""Trading bot package.

macOS fork-safety workaround (defense in depth). Any fork() after Apple's
Network framework is initialized SIGSEGVs the child in the framework's
atfork handlers (signature: "multi-threaded process forked" /
"crashed on child side of fork pre-exec" / nw_settings_child_has_forked).
The framework gets initialized by the system proxy lookup (_scproxy):

1. no_proxy="*" — makes `requests` bypass proxy lookup entirely.
2. urllib monkeypatch — anything calling urllib.request.getproxies()
   directly (streamlit, plotly, telemetry, yfinance internals) would STILL
   fall through to _scproxy when no *_proxy env vars are set. We replace
   the macOS sysconf lookup with a no-op so _scproxy is never imported.

Must run before any network library — this module is the package root, so
every `trading_bot.*` import passes through here first.
"""

import os
import sys
import types

if sys.platform == "darwin":
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("NO_PROXY", "*")

    # Stub _scproxy BEFORE urllib.request can import the real C extension —
    # the .so links SystemConfiguration/Network and its lookup calls start
    # the framework threads whose atfork handlers crash forked children.
    if "_scproxy" not in sys.modules:
        _stub = types.ModuleType("_scproxy")
        _stub._get_proxy_settings = lambda: {"exclude_simple": True}
        _stub._get_proxies = lambda: {}
        sys.modules["_scproxy"] = _stub

    import urllib.request as _urlreq

    _urlreq.getproxies_macosx_sysconf = lambda: {}
    _urlreq.proxy_bypass_macosx_sysconf = lambda host: True
    _urlreq.getproxies = _urlreq.getproxies_environment
    _urlreq.proxy_bypass = _urlreq.proxy_bypass_environment

    # 3. THE definitive layer: never fork() at all for subprocesses.
    #    Other Apple frameworks (CFNetwork via any lib, os_log, …) can still
    #    initialize Network.framework through paths we don't control; any
    #    later fork() then SIGSEGVs the child in atfork handlers. posix_spawn
    #    creates the child WITHOUT running atfork handlers — immune by
    #    construction. Python falls back to fork only for calls that
    #    posix_spawn can't express (e.g. preexec_fn — we never use it).
    import subprocess as _subprocess

    _subprocess._USE_POSIX_SPAWN = True

# Silenzia numpy RuntimeWarnings durante research/IC scan (NaN nei calcoli
# di correlazione su ticker con storia parziale — innocui, intasano il log).
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message="invalid value encountered.*",
    category=RuntimeWarning,
)
_warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice.*",
    category=RuntimeWarning,
)
_warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom .* <= 0.*",
    category=RuntimeWarning,
)
import numpy as _np

_np.seterr(invalid="ignore", divide="ignore")
