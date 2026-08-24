"""Tests for URL fetch scheme/network policy (SSRF and local file read)."""

from __future__ import annotations

from email.message import Message
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import ipaddress
import logging
import socket
import threading
import urllib.request
from unittest.mock import MagicMock, patch
from urllib.response import addinfourl

import pytest

from ag_ui.core import ImageInputContent
from ag_ui.core.types import InputContentUrlSource

from ag_ui_strands.utils import (
    UrlFetchPolicy,
    UrlFetchPolicyError,
    convert_agui_content_to_strands,
    _fetch_url_bytes,
    _validate_fetch_url,
)


def _mock_response(payload: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.side_effect = lambda n=None: payload if n is None else payload[:n]
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _addrinfo(ip: str, family: int = socket.AF_INET, port: int = 80):
    return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]


def _http_response(url: str, status: int, body: bytes = b"", location: str | None = None):
    headers = Message()
    headers["Content-Length"] = str(len(body))
    if location is not None:
        headers["Location"] = location
    response = addinfourl(BytesIO(body), headers, url, status)
    response.msg = "Found" if 300 <= status < 400 else "OK"
    return response


_MALFORMED_URLS = [
    "http://example.com:99999/a.png",
    "http://example.com:-1/a.png",
    "http://example.com:abc/a.png",
    "http://[not-an-ip]/a.png",
    "http://[::1/a.png",
    "http://[cdn.example.com]/a.png",
    "http://ex℀mple.com/a.png",
    "http://" + "a" * 64 + ".example.com/a.png",
]


# ---------------------------------------------------------------------------
# Scheme allowlist
# ---------------------------------------------------------------------------


class TestSchemeAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "file://localhost/etc/shadow",
            "ftp://example.com/secret.txt",
            "gopher://example.com:70/_test",
            "data:text/plain;base64,aGVsbG8=",
            "jar:file:///etc/passwd!/",
        ],
    )
    @patch("ag_ui_strands.utils._open_url")
    def test_non_http_schemes_are_rejected(self, mock_open, url):
        assert _fetch_url_bytes(url) is None
        mock_open.assert_not_called()

    def test_validate_raises_with_clear_message_for_file_scheme(self):
        with pytest.raises(UrlFetchPolicyError) as exc:
            _validate_fetch_url("file:///etc/passwd")
        assert "scheme" in str(exc.value).lower()
        assert "file" in str(exc.value)

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    def test_https_is_allowed(self, _mock_dns):
        assert _validate_fetch_url("https://example.com/file.txt") is None

    @patch("ag_ui_strands.utils._open_url")
    def test_private_network_opt_in_does_not_relax_scheme_allowlist(self, mock_open):
        assert (
            _fetch_url_bytes(
                "file:///etc/passwd",
                policy=UrlFetchPolicy(allow_private_networks=True),
            )
            is None
        )
        mock_open.assert_not_called()


class TestMalformedUrls:
    @pytest.mark.parametrize("url", _MALFORMED_URLS)
    @patch("ag_ui_strands.utils._open_url")
    def test_malformed_url_is_refused_without_raising(self, mock_open, url):
        assert _fetch_url_bytes(url) is None
        mock_open.assert_not_called()

    @patch("ag_ui_strands.utils._open_url")
    def test_malformed_image_url_is_dropped_during_conversion(self, mock_open):
        item = ImageInputContent(
            type="image",
            source=InputContentUrlSource(
                type="url",
                value="http://example.com:abc/a.png",
                mime_type="image/png",
            ),
        )

        assert convert_agui_content_to_strands([item]) == []
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# Private / loopback / link-local network blocking
# ---------------------------------------------------------------------------


class TestNetworkRangeBlocking:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://127.0.0.1:8080/admin",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
            "http://172.16.0.1/",
            "http://100.64.1.1/",
            "http://224.0.0.1/",
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://[ff02::1]/",
            "http://0.0.0.0/",
        ],
    )
    @patch("ag_ui_strands.utils._open_url")
    def test_private_targets_are_rejected(self, mock_open, url):
        assert _fetch_url_bytes(url) is None
        mock_open.assert_not_called()

    def test_metadata_ip_error_mentions_blocked_address(self):
        with pytest.raises(UrlFetchPolicyError) as exc:
            _validate_fetch_url("http://169.254.169.254/latest/meta-data/")
        assert "169.254.169.254" in str(exc.value)

    @patch(
        "ag_ui_strands.utils.socket.getaddrinfo",
        return_value=_addrinfo("169.254.169.254"),
    )
    def test_hostname_resolving_to_metadata_ip_is_rejected(self, _mock_dns):
        with pytest.raises(UrlFetchPolicyError):
            _validate_fetch_url("http://metadata.attacker.example/")

    @patch("ag_ui_strands.utils._open_url")
    @patch(
        "ag_ui_strands.utils.socket.getaddrinfo",
        return_value=_addrinfo("127.0.0.1"),
    )
    def test_hostname_resolving_to_loopback_is_rejected(self, _mock_dns, mock_open):
        assert _fetch_url_bytes("http://localhost:9000/") is None
        mock_open.assert_not_called()

    @patch("ag_ui_strands.utils.socket.getaddrinfo", side_effect=socket.gaierror("nope"))
    def test_unresolvable_hostname_is_rejected(self, _mock_dns):
        with pytest.raises(UrlFetchPolicyError):
            _validate_fetch_url("http://does-not-exist.invalid/")

    @patch(
        "ag_ui_strands.utils.socket.getaddrinfo",
        return_value=_addrinfo("93.184.216.34") + _addrinfo("127.0.0.1"),
    )
    def test_any_private_resolved_address_rejects(self, _mock_dns):
        """A host resolving to both a public and a private address is rejected."""
        with pytest.raises(UrlFetchPolicyError):
            _validate_fetch_url("http://rebind.example/")

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://[fe80::1]/",
        ],
    )
    @patch("ag_ui_strands.utils._open_url")
    def test_link_local_is_blocked_with_private_network_opt_in(self, mock_open, url):
        assert (
            _fetch_url_bytes(
                url,
                policy=UrlFetchPolicy(allow_private_networks=True),
            )
            is None
        )
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# DNS pinning
# ---------------------------------------------------------------------------


class TestDnsPinning:
    def test_transport_cannot_resolve_validated_hostname_again(self):
        class SecretHandler(BaseHTTPRequestHandler):
            requests = 0

            def do_GET(self):
                type(self).requests += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"loopback secret")

            def log_message(self, _format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SecretHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]

        def rebinding_dns(_host, requested_port, *args, **kwargs):
            # Policy resolution specifies IPPROTO_TCP. A second, transport-layer
            # hostname resolution instead receives the attacker's loopback answer.
            if kwargs.get("proto") == socket.IPPROTO_TCP:
                return _addrinfo("93.184.216.34", port=requested_port)
            return _addrinfo("127.0.0.1", port=requested_port)

        try:
            with (
                patch(
                    "ag_ui_strands.utils.socket.getaddrinfo",
                    side_effect=rebinding_dns,
                ),
                patch("urllib.request.getproxies", return_value={}),
            ):
                result = _fetch_url_bytes(
                    f"http://rebind.example:{port}/secret",
                    policy=UrlFetchPolicy(timeout=0.1),
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()

        assert result is None
        assert SecretHandler.requests == 0

    def test_original_hostname_is_preserved_for_host_header(self):
        class HostHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = self.headers["Host"].encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), HostHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]

        try:
            with (
                patch(
                    "ag_ui_strands.utils.socket.getaddrinfo",
                    return_value=_addrinfo("127.0.0.1", port=port),
                ),
                patch("urllib.request.getproxies", return_value={}),
            ):
                result = _fetch_url_bytes(
                    f"http://content.example:{port}/file",
                    policy=UrlFetchPolicy(
                        allow_private_networks=True,
                        timeout=1.0,
                    ),
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()

        assert result == f"content.example:{port}".encode()

    def test_https_uses_original_hostname_for_tls(self):
        from ag_ui_strands.utils import _PinnedHTTPSConnection

        context = MagicMock()
        context.wrap_socket.return_value = MagicMock()
        raw_socket = MagicMock()
        addresses = (ipaddress.ip_address("93.184.216.34"),)
        connection = _PinnedHTTPSConnection(
            "content.example",
            validated_addresses=addresses,
            context=context,
        )

        with patch(
            "ag_ui_strands.utils._connect_to_validated_addresses",
            return_value=raw_socket,
        ) as connect:
            connection.connect()

        connect.assert_called_once()
        context.wrap_socket.assert_called_once_with(
            raw_socket,
            server_hostname="content.example",
        )

    def test_pinned_connector_falls_back_between_validated_addresses(self):
        from ag_ui_strands.utils import _connect_to_validated_addresses

        connected_socket = MagicMock()
        addresses = [
            ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946"),
            ipaddress.ip_address("93.184.216.34"),
        ]

        with patch(
            "ag_ui_strands.utils.socket.socket",
            side_effect=[OSError("IPv6 unavailable"), connected_socket],
        ):
            result = _connect_to_validated_addresses(addresses, 443, timeout=1.0)

        assert result is connected_socket
        connected_socket.connect.assert_called_once_with(("93.184.216.34", 443))

    def test_fetch_opener_ignores_environment_proxies(self):
        from ag_ui_strands.utils import _open_url

        opener = MagicMock()
        with patch("ag_ui_strands.utils.urllib.request.build_opener", return_value=opener) as build:
            _open_url("https://content.example/file", 1.0, UrlFetchPolicy())

        proxy_handlers = [
            handler
            for handler in build.call_args.args
            if isinstance(handler, urllib.request.ProxyHandler)
        ]
        assert len(proxy_handlers) == 1
        assert proxy_handlers[0].proxies == {}


# ---------------------------------------------------------------------------
# URL userinfo
# ---------------------------------------------------------------------------


class TestUserinfoUrls:
    USERNAME_MARKER = "username-marker"
    PASSWORD_MARKER = "password-marker"

    def _url(self, scheme: str = "https") -> str:
        return (
            f"{scheme}://{self.USERNAME_MARKER}:{self.PASSWORD_MARKER}"
            "@content.example/file"
        )

    @patch("ag_ui_strands.utils.socket.getaddrinfo")
    def test_validate_rejects_userinfo_without_dns_or_secret_echo(self, mock_dns):
        with pytest.raises(UrlFetchPolicyError) as exc:
            _validate_fetch_url(self._url())

        mock_dns.assert_not_called()
        assert self.USERNAME_MARKER not in str(exc.value)
        assert self.PASSWORD_MARKER not in str(exc.value)

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_fetch_rejects_userinfo_before_opening(self, mock_open, mock_dns):
        assert _fetch_url_bytes(self._url()) is None
        mock_dns.assert_not_called()
        mock_open.assert_not_called()

    @pytest.mark.parametrize(
        ("scheme", "handler_name", "open_method_name"),
        [
            ("http", "_PolicyHTTPHandler", "http_open"),
            ("https", "_PolicyHTTPSHandler", "https_open"),
        ],
    )
    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    def test_policy_handler_rejects_userinfo_at_connection_boundary(
        self,
        mock_dns,
        scheme,
        handler_name,
        open_method_name,
    ):
        import ag_ui_strands.utils as utils

        handler = getattr(utils, handler_name)(UrlFetchPolicy())
        handler.do_open = MagicMock()
        req = urllib.request.Request(self._url(scheme))

        with pytest.raises(UrlFetchPolicyError) as exc:
            getattr(handler, open_method_name)(req)

        mock_dns.assert_not_called()
        handler.do_open.assert_not_called()
        assert self.USERNAME_MARKER not in str(exc.value)
        assert self.PASSWORD_MARKER not in str(exc.value)


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


class TestRedirectValidation:
    def test_redirect_to_metadata_ip_is_blocked(self):
        from ag_ui_strands.utils import _PolicyRedirectHandler

        handler = _PolicyRedirectHandler(UrlFetchPolicy())
        req = MagicMock()
        with pytest.raises(UrlFetchPolicyError):
            handler.redirect_request(
                req,
                MagicMock(),
                302,
                "Found",
                {},
                "http://169.254.169.254/latest/meta-data/",
            )

    def test_redirect_to_file_scheme_is_blocked(self):
        from ag_ui_strands.utils import _PolicyRedirectHandler

        handler = _PolicyRedirectHandler(UrlFetchPolicy())
        with pytest.raises(UrlFetchPolicyError):
            handler.redirect_request(
                MagicMock(), MagicMock(), 302, "Found", {}, "file:///etc/passwd"
            )

    @patch(
        "ag_ui_strands.utils.socket.getaddrinfo",
        return_value=_addrinfo("93.184.216.34"),
    )
    def test_real_opener_blocks_metadata_redirect_with_private_network_opt_in(
        self, _mock_dns
    ):
        start_url = "http://public.example/start"
        metadata_url = "http://169.254.169.254/latest/meta-data/"
        attempted = []

        def fake_do_open(_handler, _http_class, req, **_kwargs):
            attempted.append(req.full_url)
            if req.full_url == start_url:
                return _http_response(start_url, 302, location=metadata_url)
            return _http_response(req.full_url, 200, body=b"metadata secret")

        with patch.object(
            urllib.request.AbstractHTTPHandler,
            "do_open",
            new=fake_do_open,
        ):
            result = _fetch_url_bytes(
                start_url,
                policy=UrlFetchPolicy(allow_private_networks=True),
            )

        assert result is None
        assert attempted == [start_url]

    @patch(
        "ag_ui_strands.utils.socket.getaddrinfo",
        side_effect=[
            _addrinfo("93.184.216.34"),
            _addrinfo("93.184.216.34"),
            _addrinfo("93.184.216.34"),
            _addrinfo("127.0.0.1"),
        ],
    )
    def test_redirect_is_revalidated_at_connection_boundary(self, _mock_dns):
        start_url = "http://public.example/start"
        redirect_url = "http://rebind.example/secret"
        attempted = []

        def fake_do_open(_handler, _http_class, req, **_kwargs):
            attempted.append(req.full_url)
            if req.full_url == start_url:
                return _http_response(start_url, 302, location=redirect_url)
            return _http_response(req.full_url, 200, body=b"loopback secret")

        with patch.object(
            urllib.request.AbstractHTTPHandler,
            "do_open",
            new=fake_do_open,
        ):
            result = _fetch_url_bytes(start_url)

        assert result is None
        assert attempted == [start_url]


# ---------------------------------------------------------------------------
# Response size cap
# ---------------------------------------------------------------------------


class TestResponseSizeCap:
    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_oversized_response_is_rejected(self, mock_open, _mock_dns):
        mock_open.return_value = _mock_response(b"x" * 100)

        result = _fetch_url_bytes(
            "https://example.com/big.bin",
            policy=UrlFetchPolicy(max_bytes=10),
        )

        assert result is None

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_read_is_bounded_not_unlimited(self, mock_open, _mock_dns):
        resp = _mock_response(b"x" * 5)
        mock_open.return_value = resp

        _fetch_url_bytes("https://example.com/f.bin", policy=UrlFetchPolicy(max_bytes=1024))

        resp.read.assert_called_once_with(1025)

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_response_exactly_at_limit_is_returned(self, mock_open, _mock_dns):
        mock_open.return_value = _mock_response(b"x" * 10)

        result = _fetch_url_bytes(
            "https://example.com/exact.bin",
            policy=UrlFetchPolicy(max_bytes=10),
        )

        assert result == b"x" * 10

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_within_limit_is_returned(self, mock_open, _mock_dns):
        mock_open.return_value = _mock_response(b"hello")

        result = _fetch_url_bytes(
            "https://example.com/small.txt", policy=UrlFetchPolicy(max_bytes=1024)
        )

        assert result == b"hello"


# ---------------------------------------------------------------------------
# Safe logging
# ---------------------------------------------------------------------------


class TestUrlLogRedaction:
    @staticmethod
    def _assert_redacted(caplog, url: str, *secret_markers: str):
        expected_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        assert f"url_id={expected_id}" in caplog.text
        assert url not in caplog.text
        for marker in secret_markers:
            assert marker not in caplog.text

    def test_policy_rejection_does_not_log_raw_url(self, caplog):
        url = (
            "http://username-marker:password-marker@127.0.0.1/download"
            "?X-Amz-Credential=credential-marker"
            "&X-Amz-Signature=signature-marker#fragment-marker"
        )

        with caplog.at_level(logging.WARNING, logger="ag_ui_strands.utils"):
            result = _fetch_url_bytes(url)

        assert result is None
        self._assert_redacted(
            caplog,
            url,
            "username-marker",
            "password-marker",
            "credential-marker",
            "signature-marker",
            "fragment-marker",
        )

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_oversized_response_does_not_log_signed_url(
        self,
        mock_open,
        _mock_dns,
        caplog,
    ):
        url = (
            "https://content.example/file?X-Amz-Credential=credential-marker"
            "&X-Amz-Signature=signature-marker#fragment-marker"
        )
        mock_open.return_value = _mock_response(b"too large")

        with caplog.at_level(logging.WARNING, logger="ag_ui_strands.utils"):
            result = _fetch_url_bytes(url, policy=UrlFetchPolicy(max_bytes=2))

        assert result is None
        self._assert_redacted(
            caplog,
            url,
            "credential-marker",
            "signature-marker",
            "fragment-marker",
        )

    @patch("ag_ui_strands.utils.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34"))
    @patch("ag_ui_strands.utils._open_url")
    def test_transport_exception_text_cannot_echo_signed_url(
        self,
        mock_open,
        _mock_dns,
        caplog,
    ):
        url = "https://content.example/file?token=secret-marker#fragment-marker"
        echoed_url = (
            "https://username-marker:password-marker@content.example/echoed"
        )
        mock_open.side_effect = OSError(
            f"connection failed while requesting {echoed_url}"
        )

        with caplog.at_level(logging.WARNING, logger="ag_ui_strands.utils"):
            result = _fetch_url_bytes(url)

        assert result is None
        self._assert_redacted(
            caplog,
            url,
            "secret-marker",
            "fragment-marker",
            echoed_url,
            "username-marker",
            "password-marker",
        )


# ---------------------------------------------------------------------------
# Configurability (safe by default, opt-in relaxation)
# ---------------------------------------------------------------------------


class TestPolicyConfiguration:
    def test_defaults_are_safe(self):
        policy = UrlFetchPolicy()
        assert policy.allowed_schemes == frozenset({"http", "https"})
        assert policy.allow_private_networks is False
        assert policy.max_bytes > 0

    @patch("ag_ui_strands.utils._open_url")
    def test_private_networks_can_be_opted_into(self, mock_open):
        mock_open.return_value = _mock_response(b"local")

        result = _fetch_url_bytes(
            "http://127.0.0.1:8000/f.txt",
            policy=UrlFetchPolicy(allow_private_networks=True),
        )

        assert result == b"local"

    @patch("ag_ui_strands.utils._open_url")
    def test_extra_scheme_can_be_opted_into(self, mock_open):
        mock_open.return_value = _mock_response(b"data")

        result = _fetch_url_bytes(
            "ftp://127.0.0.1/f.txt",
            policy=UrlFetchPolicy(
                allowed_schemes=frozenset({"ftp"}), allow_private_networks=True
            ),
        )

        assert result == b"data"


# ---------------------------------------------------------------------------
# End-to-end through content conversion
# ---------------------------------------------------------------------------


class TestConversionDoesNotFetchBlockedUrls:
    @patch("ag_ui_strands.utils._open_url")
    def test_image_url_source_with_file_scheme_is_dropped(self, mock_open):
        from ag_ui.core import ImageInputContent
        from ag_ui.core.types import InputContentUrlSource

        from ag_ui_strands.utils import convert_agui_content_to_strands

        item = ImageInputContent(
            type="image",
            source=InputContentUrlSource(
                type="url", value="file:///etc/passwd", mime_type="image/png"
            ),
        )

        blocks = convert_agui_content_to_strands([item])

        assert blocks == []
        mock_open.assert_not_called()
