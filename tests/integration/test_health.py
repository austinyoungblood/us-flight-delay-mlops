import asyncio

import httpx

from services.api.app.main import app


def test_health_skeleton_reports_dependencies_unavailable() -> None:
    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    response = asyncio.run(request_health())
    assert response.status_code == 200
    assert response.json() == {
        "service": "flight-delay-api",
        "status": "healthy",
        "model_loaded": False,
        "database_connected": False,
    }


def test_only_health_business_endpoint_is_exposed() -> None:
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/predict" not in paths
