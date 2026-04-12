import base64
import importlib
import io
import json
import struct
import sys
import types
import unittest
from collections import deque

from PIL import Image


def _make_image_bytes(fmt, color):
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(buf, format=fmt)
    return buf.getvalue()


TEST_PNG = _make_image_bytes('PNG', (20, 120, 220))
TEST_JPEG = _make_image_bytes('JPEG', (220, 140, 40))
TEST_RESULT_B64 = 'data:image/jpeg;base64,' + base64.b64encode(TEST_JPEG).decode('ascii')


class FakeSiftEngine:
    def __init__(self):
        self.coord_lock_enabled = False
        self._lock_min_to_activate = 10

    def set_coord_lock(self, enabled):
        self.coord_lock_enabled = enabled
        return True


class FakeTracker:
    def __init__(self, sift_only=False):
        self.sift_engine = FakeSiftEngine()
        self.current_mode = 'sift'
        self.map_width = 1024
        self.map_height = 1024
        self.pos_history = deque(maxlen=20)
        self.latest_result_jpeg = TEST_JPEG
        self.latest_result_image = TEST_RESULT_B64
        self.latest_status = self._make_status()
        self._linear_filter_consecutive = 0
        self.last_minimap = None

    def _make_status(self):
        return {
            'mode': self.current_mode,
            'state': 'FOUND',
            'position': {'x': 128, 'y': 256},
            'found': True,
            'matches': 17,
            'match_quality': 0.93,
            'arrow_angle': 42.5,
            'arrow_stopped': False,
            'coord_lock': self.sift_engine.coord_lock_enabled,
            'hybrid': False,
            'hybrid_busy': False,
        }

    def set_minimap(self, minimap_bgr):
        self.last_minimap = minimap_bgr
        self.latest_status = self._make_status()
        self.latest_result_jpeg = TEST_JPEG
        self.latest_result_image = TEST_RESULT_B64
        if not self.pos_history:
            self.pos_history.append((128, 256))

    def process_frame(self, need_base64=True, need_jpeg=True):
        self.latest_status = self._make_status()
        return (
            self.latest_result_image if need_base64 else None,
            self.latest_result_jpeg if need_jpeg else None,
        )

    def set_mode(self, mode):
        if mode not in ('sift', 'loftr'):
            return False
        self.current_mode = mode
        self.latest_status = self._make_status()
        return True

    def get_latest_result_base64(self):
        return self.latest_result_image

    def get_latest_result_jpeg(self):
        return self.latest_result_jpeg


def _load_main_web_with_fake_tracker():
    fake_tracker_module = types.ModuleType('tracker_core')
    fake_tracker_module.AIMapTrackerWeb = FakeTracker

    original_tracker_core = sys.modules.get('tracker_core')
    sys.modules['tracker_core'] = fake_tracker_module
    sys.modules.pop('main_web', None)

    try:
        return importlib.import_module('main_web')
    finally:
        if original_tracker_core is not None:
            sys.modules['tracker_core'] = original_tracker_core
        else:
            sys.modules.pop('tracker_core', None)


MAIN_WEB = _load_main_web_with_fake_tracker()
APP = MAIN_WEB.app
SOCKETIO = MAIN_WEB.socketio


def _unpack_result_packet(packet):
    json_len = struct.unpack('>I', packet[:4])[0]
    status = json.loads(packet[4:4 + json_len].decode('utf-8'))
    jpeg_bytes = packet[4 + json_len:]
    return status, jpeg_bytes


class WebConnectionModesTest(unittest.TestCase):
    def setUp(self):
        self.http_client = APP.test_client()
        MAIN_WEB.tracker.sift_engine.coord_lock_enabled = False
        MAIN_WEB.tracker.pos_history.clear()
        MAIN_WEB.tracker.latest_result_jpeg = TEST_JPEG
        MAIN_WEB.tracker.latest_result_image = TEST_RESULT_B64
        MAIN_WEB.tracker.latest_status = MAIN_WEB.tracker._make_status()

    def test_upload_minimap_accepts_json_base64(self):
        response = self.http_client.post(
            '/api/upload_minimap',
            json={'image': 'data:image/png;base64,' + base64.b64encode(TEST_PNG).decode('ascii')},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertTrue(payload['image'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(payload['status']['position'], {'x': 128, 'y': 256})
        self.assertEqual(payload['status']['matches'], 17)

    def test_upload_minimap_accepts_form_data(self):
        response = self.http_client.post(
            '/api/upload_minimap',
            data={'image': (io.BytesIO(TEST_PNG), 'mini.png')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(payload['status']['state'], 'FOUND')

    def test_latest_frame_returns_jpeg_binary(self):
        MAIN_WEB.tracker.set_minimap('frame-ready')

        response = self.http_client.get('/api/latest_frame')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/jpeg')
        self.assertEqual(response.data, TEST_JPEG)

    def test_socket_connect_sends_initial_status(self):
        client = SOCKETIO.test_client(APP, flask_test_client=self.http_client)
        try:
            received = client.get_received()
            status_events = [event for event in received if event['name'] == 'status']
            self.assertTrue(status_events)
            self.assertEqual(status_events[0]['args'][0]['position'], {'x': 128, 'y': 256})
        finally:
            client.disconnect()

    def test_socket_frame_event_returns_binary_result_packet(self):
        client = SOCKETIO.test_client(APP, flask_test_client=self.http_client)
        try:
            client.get_received()  # 清掉 connect 时的 status 事件
            client.emit('frame', TEST_JPEG)
            received = client.get_received()
            result_events = [event for event in received if event['name'] == 'result']
            self.assertTrue(result_events)

            status, jpeg_bytes = _unpack_result_packet(result_events[-1]['args'][0])
            self.assertEqual(status['x'], 128)
            self.assertEqual(status['y'], 256)
            self.assertEqual(status['f'], 1)
            self.assertEqual(jpeg_bytes, TEST_JPEG)
        finally:
            client.disconnect()

    def test_watch_client_receives_broadcast_frame(self):
        watcher = SOCKETIO.test_client(APP, flask_test_client=APP.test_client())
        sender = SOCKETIO.test_client(APP, flask_test_client=APP.test_client())
        try:
            watcher.get_received()
            sender.get_received()

            sender.emit('frame', TEST_JPEG)
            watcher_events = watcher.get_received()
            result_events = [event for event in watcher_events if event['name'] == 'result']
            self.assertTrue(result_events)

            status, jpeg_bytes = _unpack_result_packet(result_events[-1]['args'][0])
            self.assertEqual(status['f'], 1)
            self.assertEqual(status['c'], 17)
            self.assertEqual(jpeg_bytes, TEST_JPEG)
        finally:
            sender.disconnect()
            watcher.disconnect()


if __name__ == '__main__':
    unittest.main()
