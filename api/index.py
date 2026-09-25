from app import app, home, health, chat

app.add_api_route("/api", home, methods=["GET"])
app.add_api_route("/api/health", health, methods=["GET"])
app.add_api_route("/api/chat", chat, methods=["POST"])