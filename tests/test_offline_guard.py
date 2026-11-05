"""The test suite never reaches the network (conftest's autouse fixtures)."""

import socket
import time

import pytest


def test_a_connection_beyond_loopback_is_refused_at_once():
    addr = socket.getaddrinfo("api.example.org", 443)[0][4]  # faked: a public IP
    start = time.monotonic()
    with pytest.raises(ConnectionRefusedError), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(addr)
    assert time.monotonic() - start < 1  # a real connect to that IP waits out its timeout


def test_ip_literals_and_localhost_are_not_remapped():
    # Not DNS: a closed local port must fail fast as "connection refused", not reach a public IP
    assert socket.getaddrinfo("127.0.0.1", 19999)[0][4] == ("127.0.0.1", 19999)
    assert socket.getaddrinfo("localhost", 11434)[0][4] == ("127.0.0.1", 11434)
