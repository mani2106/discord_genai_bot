import json
import os
import types
# import pytest
from types import SimpleNamespace

import image_cap_flow.llm_backends as backends


class DummyResponse:
    def __init__(self, json_obj, status=200):
        self._json = json_obj
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"status {self.status_code}")


def test_requests_embedding_single(monkeypatch):
    # Mock requests.post
    def fake_post(url, headers=None, json=None, timeout=None):
        assert 'Authorization' in headers and headers['Authorization'].startswith('Bearer ')
        # return a sample response shape
        return DummyResponse({'data': [{'embedding': [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(backends, 'requests', SimpleNamespace(post=fake_post))

    os.environ['OPENROUTER_API_KEY'] = 'test'
    os.environ['OPENROUTER_API_BASE'] = 'https://openrouter.ai'

    emb = backends.get_embed()
    assert emb is not None
    vec = emb.get_text_embedding('hello')
    assert isinstance(vec, list)
    assert vec == [0.1, 0.2, 0.3]


def test_requests_embedding_batch(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse({'data': [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr(backends, 'requests', SimpleNamespace(post=fake_post))

    os.environ['OPENROUTER_API_KEY'] = 'test'
    emb = backends.get_embed()
    out = emb.get_text_embedding_batch(['a', 'b'])
    assert isinstance(out, list)
    assert out == [[0.1, 0.2], [0.3, 0.4]]


def test_get_embed_fallback_openai(monkeypatch):
    # Simulate absence of requests POST (raise) and OpenAIEmbedding availability
    # Remove OPENROUTER key so fallback will be used
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

    class FakeOpenAIEmbedding:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(backends.__dict__, 'OpenAIEmbedding', FakeOpenAIEmbedding)

    os.environ['OPENAI_API_KEY'] = 'fake'
    emb = backends.get_embed()
    # The fallback returns an instance of FakeOpenAIEmbedding
    assert isinstance(emb, FakeOpenAIEmbedding)
