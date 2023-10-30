import json
import os

import streamlit.components.v1 as components


def network_component(
    type: str = "default",
    network: dict = None,
    max_step: int = 8,
):
    """Embeds a network component from Chromatic.

    Parameters
    ----------
    network : dict
        The network to be rendered.
    max_step : int
        The maximum number of steps in one trial.
    """

    showAllEdges = "true" if type == "legacy" else "false"
    network_args = json.dumps(network, separators=(",", ":"))

    BASE_URL = os.getenv("FRONTEND_URL", "http://localhost:9000")

    url = (
        f"{BASE_URL}/streamlit?network={network_args}&max_moves={max_step}"
        f"&showAllEdges={showAllEdges}"
    )

    components.iframe(url, height=700, width=800)
