# -*- coding: utf-8 -*-
"""
Tests for the DeepSeek Search provider (offline, HTTP layer mocked).
"""

import sys
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock newspaper before search_service import (optional dependency)
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.search_service import DeepSeekSearchProvider, SearchService


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _success_payload():
    return {
        "results": [
            {
                "rerank_score": 2.4,
                "title": "贵州茅台发布最新财报",
                "url": "https://finance.example.com/news/600519",
                "snippet": "短摘要，面向展示。",
                "summary": "长摘要，面向模型阅读。" * 80,
                "content": "",
                "site_name": "Example 财经",
                "date": "2026-06-01T00:00:00+00:00",
                "authority": 3,
            },
            {
                "title": "第二条结果（无 site_name/summary）",
                "url": "https://www.example.org/aapl",
                "snippet": "只有 snippet 的结果。",
                "date": None,
            },
        ]
    }


class TestDeepSeekSearchProvider(unittest.TestCase):
    """Tests for DeepSeek Search request building and result mapping."""

    def test_success_maps_fields_and_prefers_summary(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        fake = _FakeResponse(status_code=200, payload=_success_payload())

        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            resp = provider.search("贵州茅台 600519 最新 新闻", max_results=5, days=3)

        self.assertTrue(resp.success)
        self.assertEqual(resp.provider, "DeepSeek")
        self.assertEqual(len(resp.results), 2)

        first = resp.results[0]
        self.assertEqual(first.title, "贵州茅台发布最新财报")
        self.assertEqual(first.source, "Example 财经")
        self.assertEqual(first.published_date, "2026-06-01T00:00:00+00:00")
        self.assertEqual(first.url, "https://finance.example.com/news/600519")
        # summary 面向模型阅读，优先于 snippet，且截断到 500 字符
        self.assertTrue(first.snippet.startswith("长摘要，面向模型阅读。"))
        self.assertLessEqual(len(first.snippet), 500)

        # 第二条无 site_name → 回退域名；无 summary → 回退 snippet
        second = resp.results[1]
        self.assertEqual(second.source, "example.org")
        self.assertEqual(second.snippet, "只有 snippet 的结果。")

        # 请求契约：Bearer 鉴权 + need_content=False
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer dummy_key")
        self.assertIs(kwargs["json"]["need_content"], False)
        self.assertEqual(kwargs["json"]["count"], 5)

    def test_days_appends_after_operator(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        fake = _FakeResponse(status_code=200, payload={"results": []})

        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            provider.search("贵州茅台 财报", max_results=5, days=3)

        sent_query = mock_post.call_args.kwargs["json"]["query"]
        expected_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        self.assertEqual(sent_query, f"贵州茅台 财报 after:{expected_date}")

    def test_long_query_skips_after_operator_to_stay_within_limit(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        long_query = "长" * 120  # 原始 query 已接近 128 上限
        fake = _FakeResponse(status_code=200, payload={"results": []})

        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            provider.search(long_query, max_results=5, days=3)

        sent_query = mock_post.call_args.kwargs["json"]["query"]
        self.assertEqual(sent_query, long_query)
        self.assertLessEqual(len(sent_query), DeepSeekSearchProvider._MAX_QUERY_LENGTH)

    def test_count_clamped_to_api_maximum(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        fake = _FakeResponse(status_code=200, payload={"results": []})

        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            provider.search("贵州茅台", max_results=100, days=3)

        self.assertEqual(mock_post.call_args.kwargs["json"]["count"], 50)

    def test_error_status_codes_are_mapped(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        cases = [
            (401, {"error": {"message": "invalid key"}}, "API KEY无效"),
            (402, {"error": {"message": "insufficient balance"}}, "账户余额不足"),
            (400, {"error": {"message": "bad query"}}, "请求参数错误"),
            (429, {"error": {"message": "rate limited"}}, "请求频率达到限制"),
        ]
        for status_code, payload, expected_prefix in cases:
            with self.subTest(status_code=status_code):
                fake = _FakeResponse(status_code=status_code, payload=payload)
                with patch("src.search_service._post_with_retry", return_value=fake):
                    resp = provider.search("贵州茅台", max_results=5, days=3)
                self.assertFalse(resp.success)
                self.assertTrue(resp.error_message.startswith(expected_prefix))

    def test_non_json_error_body_falls_back_to_text(self) -> None:
        provider = DeepSeekSearchProvider(["dummy_key"])
        fake = _FakeResponse(status_code=500, payload=None, text="upstream error")

        with patch("src.search_service._post_with_retry", return_value=fake):
            resp = provider.search("贵州茅台", max_results=5, days=3)

        self.assertFalse(resp.success)
        self.assertIn("HTTP 500", resp.error_message)
        self.assertIn("upstream error", resp.error_message)


class TestDeepSeekSearchRegistration(unittest.TestCase):
    """Tests for provider registration inside SearchService."""

    def test_deepseek_keys_register_provider(self) -> None:
        service = SearchService(
            deepseek_keys=["key_a", "key_b"],
            searxng_public_instances_enabled=False,
        )
        self.assertTrue(service.is_available)
        deepseek_providers = [
            p for p in service._providers if isinstance(p, DeepSeekSearchProvider)
        ]
        self.assertEqual(len(deepseek_providers), 1)
        self.assertEqual(deepseek_providers[0].name, "DeepSeek")
        self.assertEqual(deepseek_providers[0]._api_keys, ["key_a", "key_b"])

    def test_without_deepseek_keys_no_provider_registered(self) -> None:
        service = SearchService(searxng_public_instances_enabled=False)
        self.assertFalse(
            any(isinstance(p, DeepSeekSearchProvider) for p in service._providers)
        )

    def test_constructor_kwargs_roundtrip(self) -> None:
        service = SearchService(deepseek_keys=["key_a"])
        rebuilt_kwargs = dict(service._constructor_kwargs)
        self.assertEqual(rebuilt_kwargs["deepseek_keys"], ["key_a"])
        rebuilt = SearchService(**rebuilt_kwargs)
        self.assertTrue(
            any(isinstance(p, DeepSeekSearchProvider) for p in rebuilt._providers)
        )


if __name__ == "__main__":
    unittest.main()
