import os
import sys
from random import randint
sys.path.insert(0,os.path.join(os.path.dirname(__file__), ".."))
import unittest
from src import dataset

class testDataset(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        index = randint(len(os.listdir(os.path.join(
            os.path.dirname(__file__),"..\\dataSet\\ct_scan"))))
        
        
        
        
if __name__ == "__main__":
    unittest.main()