# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import re
import subprocess
import sys
from pathlib import Path

import pytest
import requests

NOTEBOOK = Path(__file__).parents[1] / 'docs/tutorials/m_geospatial.py'
IMAGE_URLS = re.findall(r'<img src="([^"]+)"', NOTEBOOK.read_text())


def test_notebook_runs() -> None:
    subprocess.run([sys.executable, str(NOTEBOOK)], check=True)


@pytest.mark.slow
@pytest.mark.enable_socket
@pytest.mark.parametrize('url', IMAGE_URLS)
def test_image_url(url: str) -> None:
    assert (
        requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).status_code
        == 200
    )
