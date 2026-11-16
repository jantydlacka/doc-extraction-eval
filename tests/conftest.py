import os
import shutil
import socket
import tempfile

import pytest
import pytest_socket

_saved_environ: dict[str, str] | None = None
_session_home: str | None = None
_saved_socket: type | None = None


def _is_credential_var(name: str) -> bool:
    # The SDK authenticates not only via the API key but also via ANTHROPIC_PROFILE,
    # workload-identity variables and ANTHROPIC_CONFIG_DIR; a set ANTHROPIC_CONFIG_DIR
    # makes a missing profile an error, so it is removed rather than redirected.
    return name.startswith("ANTHROPIC_")


def pytest_configure(config):
    # Collection, module-level code and session fixtures run outside the per-test fixture,
    # and without pyproject addopts nothing else blocks sockets.
    global _saved_environ, _session_home, _saved_socket
    _saved_environ = dict(os.environ)
    _saved_socket = socket.socket
    _session_home = tempfile.mkdtemp(prefix="llmeval-test-home-")
    for name in [n for n in os.environ if _is_credential_var(n)]:
        del os.environ[name]
    # Profiles on disk are found under ~/.config/anthropic.
    os.environ["HOME"] = _session_home
    pytest_socket.disable_socket(allow_unix_socket=False)


def pytest_unconfigure(config):
    pytest_socket.enable_socket()
    if _saved_socket is not None:
        socket.socket = _saved_socket
    if _saved_environ is not None:
        os.environ.clear()
        os.environ.update(_saved_environ)
    if _session_home is not None:
        shutil.rmtree(_session_home, ignore_errors=True)


def pytest_runtest_setup(item):
    # pytest-socket lifts its block after every test, in its own pytest_runtest_teardown,
    # which runs before fixture teardown: fixture finalizers therefore run with sockets open.
    # Credentials stay scrubbed there, so an API call from a finalizer fails to authenticate.
    pytest_socket.disable_socket(allow_unix_socket=False)


@pytest.fixture(autouse=True)
def _no_api_credentials(monkeypatch, tmp_path_factory):
    for name in list(os.environ):
        if _is_credential_var(name):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
