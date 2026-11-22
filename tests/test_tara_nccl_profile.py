"""Fail-closed Tara NCCL profile; LL128 corrupted data on two GH200 nodes."""
import subprocess
import os
import unittest
from pathlib import Path
PROFILE = Path(__file__).resolve().parents[1] / 'examples/16_all_layers_comparison/tara_nccl_env.sh'
class TestProfile(unittest.TestCase):
    def run_profile(self, proto=None):
        env = dict(os.environ); env.pop('NCCL_PROTO', None)
        if proto is not None: env['NCCL_PROTO'] = proto
        return subprocess.run(['bash','-c','source "$1" || exit $?; printf "%s" "$NCCL_PROTO"','bash',str(PROFILE)], env=env, capture_output=True,text=True)
    def test_defaults_safe_and_visible(self):
        p=self.run_profile(); self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(p.stdout,'Simple'); self.assertIn('LL128',p.stderr)
    def test_preserves_verified_LL(self):
        p=self.run_profile('LL'); self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(p.stdout,'LL')
    def test_rejects_unsafe_protocol(self):
        p=self.run_profile('LL128'); self.assertNotEqual(p.returncode,0); self.assertIn('LL128',p.stderr)
if __name__=='__main__': unittest.main()
