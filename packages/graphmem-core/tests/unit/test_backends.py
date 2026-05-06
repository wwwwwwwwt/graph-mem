from graphmem.llm.noop import NoOpLLMClient
from graphmem.embed.sentence_transformer import SentenceTransformerEmbedClient
from graphmem.queue.sqlite_queue import SQLiteTaskQueue


def test_noop_llm_returns_empty():
    client = NoOpLLMClient()
    result = client.complete("prompt")
    assert result == ""


def test_sentence_transformer_embed():
    client = SentenceTransformerEmbedClient(model_name="all-MiniLM-L6-v2")
    vecs = client.embed(["hello world", "goodbye"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_sqlite_queue_enqueue_and_dequeue(tmp_home):
    q = SQLiteTaskQueue(str(tmp_home / "queue.db"))
    q.enqueue("compress", {"scope": "s1", "session_id": "sess1"})
    tasks = q.dequeue(limit=1)
    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "compress"
    q.close()
