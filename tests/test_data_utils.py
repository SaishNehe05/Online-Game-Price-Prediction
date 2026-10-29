import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_utils import parse_price, parse_count, clean_dataset, model_features


class DataUtilsTests(unittest.TestCase):
    def test_parse_price_european_and_free(self):
        self.assertAlmostEqual(parse_price("4,99 €"), 4.99, places=2)
        self.assertAlmostEqual(parse_price("1.234,56 €"), 1234.56, places=2)
        self.assertEqual(parse_price("Gratis"), 0.0)

    def test_parse_count(self):
        self.assertEqual(parse_count("1.234 reviews"), 1234.0)
        self.assertEqual(parse_count(45), 45.0)

    def test_clean_dataset_and_features(self):
        raw = pd.DataFrame({
            "title": ["A", "B", "C"],
            "genres": ["Action", "RPG", None],
            "price": ["4,99 €", "Gratis", "9,99 €"],
            "launch_date": ["12 Mar, 2020", "2021", "unknown"],
            "reviews": ["1.234", "0", None],
            "general_reviews": ["Positive", None, "Mixed"],
            "discount": ["0%", "0%", "20%"],
            "original_price": ["4,99 €", "Gratis", "12,49 €"],
            "is_error": [0, 0, 0],
        })
        cleaned = clean_dataset(raw)
        X, y = model_features(cleaned)
        self.assertEqual(list(X.columns), ["genres", "launch_year", "reviews", "general_reviews"])
        self.assertEqual(len(X), 3)
        self.assertEqual(float(y.iloc[1]), 0.0)
        self.assertNotIn("discount", X.columns)
        self.assertNotIn("original_price", X.columns)


if __name__ == "__main__":
    unittest.main()
