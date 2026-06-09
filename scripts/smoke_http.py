"""Stage-6 smoke test: the HTTP path (what the official agents repo speaks).

Uses Flask's test client against arc_agi.server.create_app() — same routes
as `Arcade.listen_and_serve()` without binding a port. Verifies: game list,
scorecard open, RESET, one action, scorecard close — all offline.

    .venv/bin/python scripts/smoke_http.py
"""

import json
from pathlib import Path

from arc_agi import Arcade
from arc_agi.base import OperationMode
from arc_agi.server import create_app

ENV_DIR = str(
    Path.home()
    / ".cache/kagglehub/competitions/arc-prize-2026-arc-agi-3/environment_files"
)


def check(resp, what):
    body = resp.get_json(silent=True)
    print(f"[{what}] HTTP {resp.status_code}: {json.dumps(body)[:200] if body is not None else resp.data[:200]}")
    assert resp.status_code == 200, f"{what} failed"
    return body


def main() -> None:
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=ENV_DIR,
        recordings_dir="recordings",
    )
    app, _api = create_app(arcade)
    client = app.test_client()

    check(client.get("/api/healthcheck"), "healthcheck")

    games = check(client.get("/api/games"), "games")
    assert isinstance(games, list) and len(games) == 25, f"expected 25 games, got {games}"
    ls20 = next(g for g in games if str(g.get("game_id", g)).startswith("ls20"))
    game_id = ls20["game_id"] if isinstance(ls20, dict) else ls20

    card = check(
        client.post("/api/scorecard/open", json={"tags": ["http-smoke"]}),
        "scorecard/open",
    )
    card_id = card["card_id"]

    frame = check(
        client.post("/api/cmd/RESET", json={"game_id": game_id, "card_id": card_id}),
        "cmd/RESET",
    )
    guid = frame["guid"]
    assert frame["state"] in ("NOT_FINISHED", "NOT_PLAYED")

    frame = check(
        client.post("/api/cmd/ACTION1", json={"game_id": game_id, "guid": guid}),
        "cmd/ACTION1",
    )
    assert frame["guid"] == guid

    closed = check(
        client.post("/api/scorecard/close", json={"card_id": card_id}),
        "scorecard/close",
    )
    env = closed["environments"][0] if closed.get("environments") else {}
    runs = env.get("runs", [{}])
    print(
        f"\nHTTP path OK: 25 games served, RESET+ACTION1 round-trip, "
        f"close reports actions={runs[0].get('actions')} on {env.get('id')}"
    )


if __name__ == "__main__":
    main()
