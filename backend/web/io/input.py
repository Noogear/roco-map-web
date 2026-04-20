from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Request


def decode_base64_image(b64_data: str):
    if not b64_data:
        return None
    if ',' in b64_data:
        b64_data = b64_data.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_image_from_request(req: Request):
    if 'image' in req.files:
        file = req.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if req.is_json:
        data = req.get_json(silent=True) or {}
        return decode_base64_image(data.get('image', ''))
    return None
