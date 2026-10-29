import csv
import tempfile
import unittest
from pathlib import Path

from recommender import RecommendationEngine, jaccard, read_purchases


class ReadPurchasesTests(unittest.TestCase):
    def test_normalizes_ids_and_rejects_invalid_purchase_rows(self):
        contents = (
            "customer_id,product_id,product_name,category,quantity\n"
            " C001 , P01 , Maize flour , Pantry , 2\n"
            "C002,P02,,Pantry,1\n"
            "C003,P03,Beans,Pantry,0\n"
            "C004,  ,Rice,Pantry,1\n"
            "c001,p01,Maize flour,Pantry,1\n"
            "C005,P05,Milk,Pantry,1,EXTRA\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "purchases.csv"
            path.write_text(contents, encoding="utf-8")

            valid, rejected = read_purchases(path)

        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]["customer_id"], "c001")
        self.assertEqual(valid[0]["product_id"], "p01")
        self.assertEqual(valid[0]["product_name"], "Maize flour")
        self.assertEqual(valid[0]["quantity"], "2")
        self.assertEqual(len(rejected), 4)
        self.assertTrue(all("_rejection_reason" in row for row in rejected))
        self.assertEqual(rejected[-1]["_extra_values"], ["EXTRA"])


def purchase(customer_id, product_id, name=None, category="Pantry"):
    return {
        "customer_id": customer_id,
        "product_id": product_id,
        "product_name": name or product_id.title(),
        "category": category,
        "quantity": "1",
    }


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.purchases = [
            purchase("c1", "maize"),
            purchase("c2", "maize"),
            purchase("c2", "beans"),
            purchase("c3", "maize"),
            purchase("c3", "rice", category="Grains"),
            purchase("c4", "maize"),
            purchase("c4", "sugar"),
            purchase("c4", "tea", category="Drinks"),
            purchase("c4", "tea", category="Drinks"),
        ]
        self.engine = RecommendationEngine(self.purchases)

    def test_jaccard_handles_overlap_and_empty_sets(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 1.0)
        self.assertEqual(jaccard(set(), {"a"}), 0.0)

    def test_histories_similarity_and_popularity_use_unique_products_and_customers(self):
        self.assertEqual(self.engine.history("C2"), {"maize", "beans"})
        self.assertEqual(self.engine.product_popularity["tea"], 1)
        self.assertEqual(self.engine.similarity("c1", "c2"), 0.5)
        self.assertEqual(self.engine.similarity("c1", "c1"), 1.0)
        with self.assertRaisesRegex(KeyError, "unknown customer"):
            self.engine.history("missing")

    def test_neighbours_are_filtered_and_ties_are_deterministic(self):
        self.assertEqual(
            self.engine.similar_customers("c1", k=2),
            [("c2", 0.5), ("c3", 0.5)],
        )
        self.assertEqual(
            RecommendationEngine(self.purchases, min_similarity=0.6)
            .similar_customers("c1"),
            [],
        )

    def test_candidates_include_support_and_score_combines_signals(self):
        evidence = self.engine.candidate_products("c1", k=2)
        self.assertEqual(
            evidence["beans"]["supporting_customers"],
            [{"customer_id": "c2", "similarity": 0.5}],
        )
        self.assertEqual(evidence["beans"]["support_count"], 1)
        self.assertAlmostEqual(self.engine.score(evidence["beans"]), 0.475)
        self.assertEqual(self.engine.recommend("c1", n=2)[0]["product_id"], "beans")

    def test_zero_neighbour_fallback_uses_popularity_and_recommend_all_keeps_everyone(self):
        engine = RecommendationEngine(self.purchases, min_similarity=0.6)
        self.assertEqual(
            engine.candidate_products("c1")["tea"]["supporting_customers"],
            [],
        )
        self.assertEqual(
            engine.recommend("c1", n=1)[0]["product_id"],
            "beans",
        )
        rows = engine.recommend_all(n=3)
        self.assertEqual({row["customer_id"] for row in rows}, {"c1", "c2", "c3", "c4"})

    def test_customer_who_bought_the_entire_catalog_gets_no_product_row(self):
        engine = RecommendationEngine(
            [purchase("solo", "beans"), purchase("solo", "rice")]
        )
        self.assertEqual(engine.recommend("solo"), [])
        self.assertEqual(
            engine.recommend_all(),
            [
                {
                    "customer_id": "solo",
                    "rank": "",
                    "product_id": "",
                    "product_name": "",
                    "category": "",
                    "score": "",
                    "supporting_customers": "",
                }
            ],
        )

    def test_unknown_customer_raises_key_error_and_export_includes_empty_results(self):
        with self.assertRaisesRegex(KeyError, "unknown customer"):
            self.engine.recommend("missing")

        rows = self.engine.recommend_all(n=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "recommendations.csv"
            self.engine.export_csv(rows, path)

            with path.open(newline="", encoding="utf-8") as csvfile:
                exported = list(csv.DictReader(csvfile))

        self.assertEqual(
            list(exported[0]),
            [
                "customer_id",
                "rank",
                "product_id",
                "product_name",
                "category",
                "score",
                "supporting_customers",
            ],
        )
        self.assertEqual({row["customer_id"] for row in exported}, {"c1", "c2", "c3", "c4"})
        self.assertTrue(all(row["product_id"] == "" for row in exported))


if __name__ == "__main__":
    unittest.main()
