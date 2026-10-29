import argparse
from pathlib import Path

from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases


def print_similarity_matrix(engine: RecommendationEngine, limit: int = 5) -> None:
    """Print a small deterministic similarity matrix for a quick sanity check."""
    customer_ids = sorted(engine.customer_products)[:limit]
    if not customer_ids:
        print("No customers available for the similarity matrix.")
        return

    column_width = max(10, *(len(customer_id) for customer_id in customer_ids))
    print("\nSimilarity matrix (first five customers):")
    print(f"{'customer_id':<{column_width}} " + " ".join(
        f"{customer_id:>{column_width}}" for customer_id in customer_ids
    ))
    for customer_a in customer_ids:
        values = " ".join(
            f"{engine.similarity(customer_a, customer_b):>{column_width}.3f}"
            for customer_b in customer_ids
        )
        print(f"{customer_a:<{column_width}} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate explainable grocery recommendations from purchase history."
    )
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Purchase CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Recommendation CSV path.",
    )
    parser.add_argument("--count", type=int, default=3, help="Recommendations per customer.")
    args = parser.parse_args()

    valid, rejected = read_purchases(args.input)
    engine = RecommendationEngine(valid)
    print_similarity_matrix(engine)

    rows = engine.recommend_all(n=args.count)
    engine.export_csv(rows, args.output)
    recommendation_count = sum(bool(row["product_id"]) for row in rows)
    print(
        "\nRun summary: "
        f"{len(valid)} valid purchase rows, {len(rejected)} rejected rows, "
        f"{len(engine.customer_products)} customers, {recommendation_count} recommendations."
    )
    print(f"Recommendations written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
