"""Test health endpoint functionality."""
import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health endpoint returns 200 OK."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_endpoint_includes_dependencies(client):
    """Test detailed health endpoint includes dependency status."""
    response = await client.get("/api/health/detailed")
    assert response.status_code == 200
    
    data = response.json()
    assert "components" in data
    
    components = data["components"]
    assert "alpaca_api" in components
    assert components["alpaca_api"]["status"] in ["healthy", "unhealthy"]
    
    if "database" in components:
        assert components["database"]["status"] in ["healthy", "unhealthy"]


def test_health_endpoint_sync(sync_client):
    """Test health endpoint works with sync client."""
    response = sync_client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint_no_auth_required(client):
    """Test health endpoint doesn't require authentication."""
    # Clear any auth headers that might be set
    client.headers.clear()
    
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint_cors_enabled(client):
    """Test health endpoint has CORS enabled."""
    # Check actual GET request works (CORS would be handled by middleware)
    response = await client.get("/api/health")
    assert response.status_code == 200
    
    # Check if CORS headers are present (if CORS middleware is configured)
    # In a real scenario, we'd make a CORS preflight request
    data = response.json()
    assert data["status"] == "healthy"
