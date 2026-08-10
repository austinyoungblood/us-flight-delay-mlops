"""Health-only FastAPI skeleton for Brief 01."""

from fastapi import FastAPI

from flight_delay.contracts import HealthResponse

app = FastAPI(
    title="U.S. Flight Delay API",
    description="Pre-departure flight-delay API scaffold; prediction is not implemented yet.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report process health without claiming unavailable dependencies are ready."""

    return HealthResponse(model_loaded=False, database_connected=False)
