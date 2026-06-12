"""
server.py — WebSocket and HTTP servers for ACVAS.

WebSocket server (port 8765): broadcasts environment events to all connected
phone dashboards and receives manual volume-override commands.

HTTP server (port 8766): serves the static/index.html dashboard file.
"""

import asyncio
import json
import http.server
import os
import threading
from functools import partial

import websockets

import actuator


# Global set of connected WebSocket client objects
CLIENTS: set = set()


async def broadcast(message: dict) -> None:
    """Send a JSON message to all connected WebSocket clients.

    Disconnected clients are silently removed from the CLIENTS set.
    """
    if not CLIENTS:
        return

    payload = json.dumps(message)
    disconnected = set()

    for ws in CLIENTS:
        try:
            await ws.send(payload)
        except websockets.ConnectionClosed:
            disconnected.add(ws)

    CLIENTS -= disconnected


async def ws_handler(websocket) -> None:
    """Handle a single WebSocket connection lifecycle.

    On connect: adds to CLIENTS.
    On message: if ``type == "override"``, dispatches volume change.
    On disconnect: removes from CLIENTS.
    """
    CLIENTS.add(websocket)
    print(f"[server] WebSocket client connected ({len(CLIENTS)} total)")

    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "override":
                target_volume = float(msg.get("volume", 0.5))
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, actuator.set_volume, target_volume,
                )
                print(f"[server] Manual override → volume {target_volume:.0%}")
    except websockets.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"[server] WebSocket client disconnected ({len(CLIENTS)} total)")


def _run_http_server(port: int) -> None:
    """Start a blocking HTTP server in a separate thread.

    Serves files from the ``static/`` directory relative to this script.
    """
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=static_dir,
    )
    httpd = http.server.HTTPServer(("0.0.0.0", port), handler)
    print(f"[server] HTTP dashboard serving on http://0.0.0.0:{port}")
    httpd.serve_forever()


async def start_server(config: dict) -> None:
    """Start the WebSocket server and the HTTP file server.

    The HTTP server runs in a daemon thread so it doesn't block the
    asyncio event loop.
    """
    http_port = config["http_port"]
    ws_port = config["ws_port"]

    # Start HTTP server in a background daemon thread
    http_thread = threading.Thread(
        target=_run_http_server,
        args=(http_port,),
        daemon=True,
    )
    http_thread.start()

    # Start WebSocket server (runs as an asyncio task)
    async with websockets.serve(ws_handler, "0.0.0.0", ws_port):
        print(f"[server] WebSocket server listening on ws://0.0.0.0:{ws_port}")
        await asyncio.Future()  # Run forever
