"""Local browser app: python -m src.simulator_app --demo (or --artifact FILE)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from .prediction import RatingState, demo_artifact, load_artifact
from .tournament import add_result, new_session, replay_session, simulate


class LocalHTTPServer(HTTPServer):
    """Do not let demo and real-model servers share one Windows address."""

    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class TournamentService:
    def __init__(self, artifact, session_path=None):
        self.artifact = copy.deepcopy(artifact)
        self.session_path = Path(session_path) if session_path else None
        self.session = None
        if self.session_path and self.session_path.exists():
            candidate = json.loads(self.session_path.read_text(encoding="utf-8"))
            replay_session(self.artifact, candidate)
            self.session = candidate

    def save(self, candidate):
        replay_session(self.artifact, candidate)
        if self.session_path:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.session_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, self.session_path)
        self.session = copy.deepcopy(candidate)

    def status(self):
        state, details = RatingState.from_dict(self.artifact["state"]), []
        if self.session:
            state, details, _ = replay_session(self.artifact, self.session)
        return {"kind": self.artifact["kind"], "cutoff": self.artifact["cutoff_exclusive"],
                "model_trained_through": self.artifact["model_trained_through"],
                "last_history_timestamp": self.artifact["last_history_timestamp"],
                "metadata": self.artifact["metadata"], "teams": sorted(state.global_ratings),
                "homes": state.homes, "series": details, "session": self.session}

    def action(self, route, payload):
        if route == "/api/bracket":
            if self.session and self.session["events"] and payload.get("replace_results") is not True:
                raise ValueError("Confirm clearing entered results before replacing the bracket.")
            self.save(new_session(self.artifact, payload["bracket"]))
        elif route == "/api/import":
            self.save(payload["session"])
        elif route == "/api/game":
            if not self.session:
                raise ValueError("Save a bracket first.")
            self.save(add_result(self.artifact, self.session, event_id=payload.get("event_id", uuid4().hex),
                                 series=payload["series"], winner=payload["winner"], timestamp=payload["timestamp"]))
        elif route == "/api/undo":
            if not self.session or not self.session["events"]:
                raise ValueError("There is no actual game to undo.")
            candidate = copy.deepcopy(self.session)
            candidate["events"].pop()
            self.save(candidate)
        elif route == "/api/simulate":
            if not self.session:
                raise ValueError("Save a bracket first.")
            return simulate(self.artifact, self.session, payload.get("simulations", 100_000), payload.get("seed", 2026))
        else:
            raise ValueError("Unknown action.")
        return self.status()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, mime="application/json"):
            encoded = json.dumps(data, allow_nan=False).encode() if mime == "application/json" else data.encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, (Path(__file__).resolve().parent.parent / "web/simulator.html").read_text(encoding="utf-8"), "text/html")
            elif self.path == "/api/status":
                self.respond(200, service.status())
            elif self.path == "/api/session" and service.session:
                self.respond(200, service.session)
            else:
                self.respond(404, {"error": "Not found."})

        def do_POST(self):
            # JSON-only requests and loopback binding keep this a local prototype.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.respond(415, {"error": "Send JSON."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Request size must be between 1 and 1,000,000 bytes.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object.")
                self.respond(200, service.action(self.path, payload))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.respond(400, {"error": str(exc)})
            except OSError:
                self.respond(500, {"error": "Could not save the session. No result was applied."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--demo", action="store_true", help="Clearly hypothetical equal-strength example; no real model or participants.")
    choice.add_argument("--artifact", help="Portable JSON package from src.export_artifact.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--session", help="Session JSON to save/resume; must belong to this exact artifact.")
    args = parser.parse_args()
    try:
        artifact = demo_artifact() if args.demo else load_artifact(args.artifact)
        service = TournamentService(artifact, args.session or (None if args.demo else "data/processed/simulator_sessions/current.json"))
        if args.demo and service.session is None:
            teams = list(artifact["state"]["global_ratings"])
            service.save(new_session(artifact, {"label": "Hypothetical equal-strength demonstration", "quarterfinals": [teams[i:i+2] for i in range(0, 8, 2)]}))
        server = LocalHTTPServer(("127.0.0.1", args.port), handler_for(service))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Cannot start simulator: {exc}\n")
    print(f"Local simulator: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
