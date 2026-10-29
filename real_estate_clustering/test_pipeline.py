"""Regression checks for cleaning, distance units, and undefined metric cases."""
import unittest
import numpy as np
import pandas as pd
from main import clean_data, make_features, evaluate, fit_dbscan, EARTH_KM


class PipelineTests(unittest.TestCase):
    def frame(self):
        n = 40
        return pd.DataFrame({
            "longitude": np.linspace(-122, -121, n), "latitude": np.linspace(36, 37, n),
            "housing_median_age": np.full(n, 20), "total_rooms": np.arange(n) + 200.,
            "total_bedrooms": np.full(n, 50.), "population": np.full(n, 300.),
            "households": np.full(n, 100.), "median_income": np.full(n, 4.),
            "median_house_value": np.arange(n) * 1000. + 100000.,
        })

    def test_cleaning_and_ratios(self):
        raw = self.frame()
        raw.loc[0, "total_bedrooms"] = np.nan
        raw.loc[1, "households"] = 0
        raw.loc[2, "latitude"] = 0
        raw = pd.concat([raw, raw.iloc[[3]]], ignore_index=True)
        clean, rejected, audit = clean_data(raw)
        self.assertEqual(len(clean), 38)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(audit["duplicate_rows_removed"], 1)
        self.assertEqual(audit["bedrooms_imputed"], 1)
        self.assertAlmostEqual(clean.iloc[0].rooms_per_household, 2.)
        self.assertTrue(np.isfinite(clean.select_dtypes("number").to_numpy()).all())

    def test_missing_schema_and_all_missing_bedrooms(self):
        with self.assertRaises(ValueError):
            clean_data(self.frame().drop(columns="latitude"))
        raw = self.frame()
        raw["total_bedrooms"] = np.nan
        with self.assertRaises(ValueError):
            clean_data(raw)

    def test_feature_price_switch(self):
        clean, _, _ = clean_data(self.frame())
        spatial, combined, metadata = make_features(clean)
        _, without_price, _ = make_features(clean, include_price=False)
        self.assertEqual(spatial.shape, (40, 2))
        self.assertEqual(combined.shape, (40, 8))
        self.assertEqual(without_price.shape, (40, 7))
        self.assertIn("median_house_value", metadata["attributes"])
        self.assertTrue(np.isfinite(combined).all())

    def test_noise_and_single_cluster_metrics(self):
        X = np.arange(12).reshape(6, 2)
        for labels, clusters, coverage in [([-1] * 6, 0, 0), ([0] * 6, 1, 1)]:
            scores = evaluate(X, np.array(labels))
            self.assertEqual(scores["n_clusters"], clusters)
            self.assertEqual(scores["coverage"], coverage)
            self.assertTrue(np.isnan(scores["silhouette"]))

    def test_haversine_kilometers(self):
        # First pair ~111 m apart, third point ~111 km away.
        coords = np.radians([[36., -120.], [36.001, -120.], [37., -120.]])
        labels = fit_dbscan(coords, 1 / EARTH_KM, 2, "haversine")
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], -1)
        self.assertEqual(labels[2], -1)


if __name__ == "__main__":
    unittest.main()
