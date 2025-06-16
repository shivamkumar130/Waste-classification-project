import unittest
import os
from tensorflow.keras.models import load_model

class TestWasteClassifier(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(os.path.exists('waste_classifier_vgg16.keras'))

    def test_model_loads(self):
        model = load_model('waste_classifier_vgg16.keras')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()