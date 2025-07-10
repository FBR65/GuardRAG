import pytest
from httpx import AsyncClient
from httpx import ASGITransport


@pytest.fixture
async def async_client():
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
