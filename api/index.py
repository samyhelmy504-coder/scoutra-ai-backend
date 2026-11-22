
from app import (
    app,
    home,
    health,
    chat,
    get_account_requests,
    approve_account_request,
    reject_account_request,
)

# Main API routes
app.add_api_route("/api", home, methods=["GET"])
app.add_api_route("/api/health", health, methods=["GET"])
app.add_api_route("/api/chat", chat, methods=["POST"])

# Account management routes
app.add_api_route(
    "/api/admin/account-requests",
    get_account_requests,
    methods=["GET"],
)

app.add_api_route(
    "/api/admin/account-requests/{request_uid}/approve",
    approve_account_request,
    methods=["POST"],
)

app.add_api_route(
    "/api/admin/account-requests/{request_uid}/reject",
    reject_account_request,
    methods=["POST"],
)
