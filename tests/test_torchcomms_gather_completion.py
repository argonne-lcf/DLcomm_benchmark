"""Native gather completion regression; hardware repro Tara job11734."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

p = Path(__file__).resolve().parents[1] / 'dl_comm/comm/torchcomms_backend.py'
spec = importlib.util.spec_from_file_location('gather_backend', p)
tcb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcb)

class Work:
    def __init__(self, events): self.events = events
    def wait(self): self.events.append('wait')

class Comm:
    def __init__(self): self.events = []; self.work = Work(self.events)
    def get_rank(self): return 0
    def get_size(self): return 2
    def gather(self, outputs, tensor, root, async_op):
        self.events.append(('native', root, async_op, outputs))
        return self.work

class TestGatherCompletion(unittest.TestCase):
    def test_blocking_cuda_gather_waits_native_async_and_drains_device(self):
        comm = Comm()
        dist = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        dist._comm = comm
        device = SimpleNamespace(type='cuda')
        tensor = SimpleNamespace(device=device)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: comm.events.append(('sync', d))))
        with patch.dict('sys.modules', {'torch': torch}):
            self.assertIsNone(dist.gather(tensor, ['a','b']))
        self.assertEqual(comm.events, [('native', 0, True, ['a','b']), 'wait', ('sync', device)])

    def test_async_cuda_gather_does_not_wait_eagerly(self):
        comm = Comm(); dist = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist); dist._comm = comm
        work = dist.gather(SimpleNamespace(device=SimpleNamespace(type='cuda')), async_op=True)
        self.assertEqual(comm.events, [('native',0,True,[])])
        work.wait()
        self.assertEqual(comm.events[-1], 'wait')

if __name__ == '__main__': unittest.main()
