from pathlib import Path

from shiny import App
from ui import app_ui
from server import server

app = App(
    app_ui,
    server,
    static_assets=Path(__file__).parent / "www"
)