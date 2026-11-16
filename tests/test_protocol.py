import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripturi'))
from summarize_results import pinball,dm_hac
class ProtocolTests(unittest.TestCase):
    def test_pinball(self):
        y=np.array([10.,10.]);p=np.array([8.,12.])
        self.assertTrue(np.allclose(pinball(y,p,.9),[1.8,.2]))
    def test_dm_identical(self):
        self.assertTrue(np.isnan(dm_hac(np.zeros(30),4)))
    def test_horizon_steps(self):
        from data import HORIZONS
        self.assertEqual(HORIZONS,{'24h':96,'7d':672,'14d':1344})
if __name__=='__main__':unittest.main()
