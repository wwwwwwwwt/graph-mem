from graphmem.embed.voyage_client import VoyageEmbedClient


def test_dim():
    client = VoyageEmbedClient(api_key="test", model="voyage-3")
    assert client.dim == 1024

    client2 = VoyageEmbedClient(api_key="test", model="voyage-3-lite")
    assert client2.dim == 512
