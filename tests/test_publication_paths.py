"""CPU-only regression checks for portable publication resource guards."""
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from attncopy.paths import gpu_lock_files, require_authorized_host


class PublicationResourceGuards(unittest.TestCase):
    def test_host_requires_explicit_matching_authorization(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_authorized_host()
        with patch.dict(os.environ, {'ATTNCOPY_ALLOWED_HOST': 'unrelated-host'}, clear=True):
            with self.assertRaises(RuntimeError):
                require_authorized_host()
        with patch.dict(os.environ, {'ATTNCOPY_ALLOWED_HOST': socket.gethostname()}, clear=True):
            require_authorized_host()

    def test_shared_locks_are_required_and_not_created_or_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / 'missing'
            with patch.dict(os.environ, {'ATTNCOPY_GPU_LOCK_DIR': str(missing)}, clear=True):
                with self.assertRaises(RuntimeError):
                    gpu_lock_files(root, 7)
            self.assertFalse(missing.exists())
            existing = root / 'allocated-host-gpu7.lock'
            existing.write_bytes(b'unchanged existing coordination file')
            with patch.dict(os.environ, {'ATTNCOPY_GPU_LOCK_DIR': str(root),
                                        'ATTNCOPY_GPU_LOCK_HOST': 'allocated-host'}, clear=True):
                self.assertEqual(gpu_lock_files(root, 7), [existing])
            self.assertEqual(existing.read_bytes(), b'unchanged existing coordination file')
