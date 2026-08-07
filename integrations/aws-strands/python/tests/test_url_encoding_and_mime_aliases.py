"""Tests for non-ASCII URL encoding and MIME type alias resolution."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ag_ui_strands.utils import _fetch_url_bytes, _mime_to_format


# ---------------------------------------------------------------------------
# _fetch_url_bytes — non-ASCII URL encoding
# ---------------------------------------------------------------------------


class TestFetchUrlBytesEncoding:
    """Verify that URLs with non-ASCII characters are percent-encoded."""

    @patch("ag_ui_strands.utils.urllib.request.urlopen")
    def test_ascii_url_unchanged(self, mock_urlopen):
        """A plain ASCII URL should be passed through without modification."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"data"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _fetch_url_bytes("https://example.com/path/file.txt")

        assert result == b"data"
        called_url = mock_urlopen.call_args[0][0]
        assert called_url == "https://example.com/path/file.txt"

    @patch("ag_ui_strands.utils.urllib.request.urlopen")
    def test_chinese_filename_is_percent_encoded(self, mock_urlopen):
        """Chinese characters in the URL path must be percent-encoded."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"content"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        url = "https://cdn.example.com/docs/大模型学习路线.txt"
        result = _fetch_url_bytes(url)

        assert result == b"content"
        called_url = mock_urlopen.call_args[0][0]
        # The Chinese characters should be percent-encoded
        assert "大模型" not in called_url
        assert "%E5%A4%A7%E6%A8%A1%E5%9E%8B" in called_url
        # .txt extension should remain intact
        assert called_url.endswith(".txt")

    @patch("ag_ui_strands.utils.urllib.request.urlopen")
    def test_already_encoded_url_not_double_encoded(self, mock_urlopen):
        """Percent-encoded sequences like %20 should not be re-encoded."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"ok"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        url = "https://example.com/my%20file.txt"
        _fetch_url_bytes(url)

        called_url = mock_urlopen.call_args[0][0]
        # %20 should remain as %20, not become %2520
        assert "%20" in called_url
        assert "%2520" not in called_url

    @patch("ag_ui_strands.utils.urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen):
        """Network errors should return None, not raise."""
        mock_urlopen.side_effect = OSError("Connection refused")

        result = _fetch_url_bytes("https://example.com/file.txt")

        assert result is None


# ---------------------------------------------------------------------------
# _mime_to_format — alias resolution
# ---------------------------------------------------------------------------


class TestMimeToFormatAliases:
    """Verify that common MIME subtypes are resolved via the alias table."""

    def test_text_plain_maps_to_txt(self):
        allowed = {"txt", "pdf", "csv", "doc", "docx", "html", "md", "xls", "xlsx"}
        assert _mime_to_format("text/plain", allowed) == "txt"

    def test_text_x_markdown_maps_to_md(self):
        allowed = {"txt", "pdf", "csv", "doc", "docx", "html", "md", "xls", "xlsx"}
        assert _mime_to_format("text/x-markdown", allowed) == "md"

    def test_application_msword_maps_to_doc(self):
        allowed = {"txt", "pdf", "csv", "doc", "docx", "html", "md", "xls", "xlsx"}
        assert _mime_to_format("application/msword", allowed) == "doc"

    def test_application_vnd_ms_excel_maps_to_xls(self):
        allowed = {"txt", "pdf", "csv", "doc", "docx", "html", "md", "xls", "xlsx"}
        assert _mime_to_format("application/vnd.ms-excel", allowed) == "xls"

    def test_direct_match_still_works(self):
        """MIME types that directly match (e.g. image/png) bypass the alias table."""
        allowed = {"png", "jpeg", "gif", "webp"}
        assert _mime_to_format("image/png", allowed) == "png"

    def test_application_pdf_still_works(self):
        allowed = {"pdf", "txt", "csv"}
        assert _mime_to_format("application/pdf", allowed) == "pdf"

    def test_unsupported_mime_returns_none(self):
        allowed = {"png", "jpeg"}
        assert _mime_to_format("application/octet-stream", allowed) is None

    def test_none_mime_returns_none(self):
        allowed = {"txt", "pdf"}
        assert _mime_to_format(None, allowed) is None

    def test_empty_string_mime_returns_none(self):
        allowed = {"txt", "pdf"}
        assert _mime_to_format("", allowed) is None
