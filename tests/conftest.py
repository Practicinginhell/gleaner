import asyncio
import inspect
import ipaddress
import socket

import pytest

# A global (public) address, so the fail-closed SSRF check in gleaner.security accepts it.
_FAKE_PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch):
    """Tests never do real DNS: every hostname resolves to one public IP.

    IP literals and localhost aren't DNS and stay as they are, so a closed local port fails
    fast. Tests that need other DNS behavior patch `socket.getaddrinfo` themselves; their
    patch wins.
    """

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        try:
            addr = str(ipaddress.ip_address(host))
        except ValueError:
            addr = "127.0.0.1" if host == "localhost" else _FAKE_PUBLIC_IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test reaches the network: connecting beyond loopback is refused at once.

    The fake DNS above points at a real public IP (example.com's); since the `llama`
    extra is installed, a model call there waited out its connect timeout per retry.
    """
    real_connect = socket.socket.connect

    def _connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
            raise ConnectionRefusedError(f"tests are offline: refused {address[0]}:{address[1]}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", _connect)


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        asyncio.run(pyfuncitem.obj(*[pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames]))
        return True
