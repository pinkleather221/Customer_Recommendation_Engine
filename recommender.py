import csv
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "purchases.csv"
OUTPUT_PATH = PROJECT_DIR / "output" / "recommendations.csv"


def read_purchases(
    path: str | Path = DATA_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load normalized purchase rows and rows rejected with a reason."""
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    with Path(path).open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError("Purchase CSV must contain a header row.")

        headers = [header.strip().casefold() for header in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise ValueError("Purchase CSV contains duplicate column names.")
        reader.fieldnames = headers
        required_headers = {"customer_id", "product_id", "product_name"}
        missing_headers = required_headers - set(headers)
        if missing_headers:
            names = ", ".join(sorted(missing_headers))
            raise ValueError(f"Purchase CSV is missing required columns: {names}.")

        for row in reader:
            rejection_reason: str | None = None
            if None in row:
                rejection_reason = "row contains more values than the header"

            customer_id = (row.get("customer_id") or "").strip().casefold()
            product_id = (row.get("product_id") or "").strip().casefold()
            product_name = (row.get("product_name") or "").strip()
            category = (row.get("category") or "").strip()
            quantity_value = row.get("quantity")

            if not rejection_reason and not customer_id:
                rejection_reason = "customer_id is blank"
            elif not rejection_reason and not product_id:
                rejection_reason = "product_id is blank"
            elif not rejection_reason and not product_name:
                rejection_reason = "product_name is blank"

            if not rejection_reason and "quantity" in headers:
                if quantity_value is None or not quantity_value.strip():
                    rejection_reason = "quantity is blank"
                else:
                    try:
                        quantity = int(quantity_value.strip())
                    except ValueError:
                        rejection_reason = "quantity must be a positive whole number"
                    else:
                        if quantity <= 0:
                            rejection_reason = "quantity must be a positive whole number"
            else:
                quantity = 1

            normalized_row = {
                key: value
                for key, value in row.items()
                if key is not None
            }
            if None in row:
                normalized_row["_extra_values"] = row[None]
            normalized_row.update(
                {
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "product_name": product_name,
                    "category": category,
                    "quantity": str(quantity) if not rejection_reason else quantity_value,
                }
            )

            if rejection_reason:
                normalized_row["_rejection_reason"] = rejection_reason
                rejected_rows.append(normalized_row)
            else:
                valid_rows.append(normalized_row)

    return valid_rows, rejected_rows


def jaccard(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity, treating two empty histories as identical."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class RecommendationEngine:
    """Recommend products from similar customers' distinct purchase histories."""

    CSV_COLUMNS = (
        "customer_id",
        "rank",
        "product_id",
        "product_name",
        "category",
        "score",
        "supporting_customers",
    )

    def __init__(
        self,
        purchases: list[dict[str, Any]],
        min_similarity: float = 0.0,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0.")

        self.min_similarity = min_similarity
        self.customer_products: dict[str, set[str]] = {}
        self.product_info: dict[str, dict[str, str]] = {}
        self.product_popularity: dict[str, int] = {}

        for purchase in purchases:
            customer_id = self._normalize_id(purchase.get("customer_id"))
            product_id = self._normalize_id(purchase.get("product_id"))
            products = self.customer_products.setdefault(customer_id, set())
            is_new_customer_product = product_id not in products
            products.add(product_id)

            if is_new_customer_product:
                self.product_popularity[product_id] = (
                    self.product_popularity.get(product_id, 0) + 1
                )

            product_name = str(purchase.get("product_name") or "").strip()
            category = str(purchase.get("category") or "").strip()
            info = self.product_info.setdefault(
                product_id,
                {"name": product_name, "category": category},
            )
            if not info["name"] and product_name:
                info["name"] = product_name
            if not info["category"] and category:
                info["category"] = category

    @staticmethod
    def _normalize_id(value: Any) -> str:
        normalized = str(value or "").strip().casefold()
        if not normalized:
            raise ValueError("Purchase customer_id and product_id must not be blank.")
        return normalized

    @staticmethod
    def _validate_count(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    def _require_customer(self, customer_id: str) -> str:
        normalized_id = self._normalize_id(customer_id)
        if normalized_id not in self.customer_products:
            raise KeyError(f"unknown customer_id: {customer_id!r}")
        return normalized_id

    def history(self, customer_id: str) -> set[str]:
        """Return a copy of a customer's product history."""
        normalized_id = self._require_customer(customer_id)
        return set(self.customer_products[normalized_id])

    def similarity(self, customer_a: str, customer_b: str) -> float:
        """Return the Jaccard similarity of two known customers."""
        normalized_a = self._require_customer(customer_a)
        normalized_b = self._require_customer(customer_b)
        return jaccard(
            self.customer_products[normalized_a],
            self.customer_products[normalized_b],
        )

    def similar_customers(
        self,
        customer_id: str,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return up to k other customers, highest similarity first."""
        self._validate_count(k, "k")
        normalized_id = self._require_customer(customer_id)
        neighbours = [
            (other_id, jaccard(self.customer_products[normalized_id], products))
            for other_id, products in self.customer_products.items()
            if other_id != normalized_id
        ]
        eligible = [
            (other_id, similarity)
            for other_id, similarity in neighbours
            if similarity >= self.min_similarity
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        return eligible[:k]

    def candidate_products(
        self,
        customer_id: str,
        k: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Collect unseen products and the neighbours (if any) supporting them."""
        self._validate_count(k, "k")
        normalized_id = self._require_customer(customer_id)
        history = self.customer_products[normalized_id]
        neighbours = self.similar_customers(normalized_id, k)
        evidence: dict[str, dict[str, Any]] = {}

        if neighbours:
            for neighbour_id, similarity in neighbours:
                for product_id in self.customer_products[neighbour_id] - history:
                    product_evidence = evidence.setdefault(
                        product_id,
                        {
                            "product_id": product_id,
                            "product_name": (
                                self.product_info[product_id]["name"] or product_id
                            ),
                            "category": self.product_info[product_id]["category"],
                            "supporting_customers": [],
                            "support_count": 0,
                            "neighbor_count": len(neighbours),
                            "popularity": self.product_popularity[product_id],
                        },
                    )
                    product_evidence["supporting_customers"].append(
                        {"customer_id": neighbour_id, "similarity": similarity}
                    )
                    product_evidence["support_count"] += 1
        else:
            for product_id in self.product_info:
                if product_id in history:
                    continue
                evidence[product_id] = {
                    "product_id": product_id,
                    "product_name": self.product_info[product_id]["name"] or product_id,
                    "category": self.product_info[product_id]["category"],
                    "supporting_customers": [],
                    "support_count": 0,
                    "neighbor_count": 0,
                    "popularity": self.product_popularity[product_id],
                }

        return {product_id: evidence[product_id] for product_id in sorted(evidence)}

    def score(self, evidence: dict[str, Any]) -> float:
        """Combine mean neighbour similarity, neighbour support, and popularity."""
        supporters = evidence["supporting_customers"]
        similarity_component = (
            sum(supporter["similarity"] for supporter in supporters) / len(supporters)
            if supporters
            else 0.0
        )
        neighbor_count = evidence["neighbor_count"]
        support_component = (
            evidence["support_count"] / neighbor_count if neighbor_count else 0.0
        )
        max_popularity = max(self.product_popularity.values(), default=0)
        popularity_component = (
            evidence["popularity"] / max_popularity if max_popularity else 0.0
        )
        return (
            0.7 * similarity_component
            + 0.2 * support_component
            + 0.1 * popularity_component
        )

    def recommend(
        self,
        customer_id: str,
        n: int = 3,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to n ranked product recommendations for a known customer."""
        self._validate_count(n, "n")
        self._validate_count(k, "k")
        normalized_id = self._require_customer(customer_id)
        candidates = self.candidate_products(normalized_id, k)
        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                -self.score(item),
                -item["support_count"],
                -item["popularity"],
                item["product_id"],
            ),
        )
        results = []
        for rank, evidence in enumerate(ranked[:n], start=1):
            supporters = "; ".join(
                f'{item["customer_id"]} ({item["similarity"]:.3f})'
                for item in evidence["supporting_customers"]
            )
            results.append(
                {
                    "customer_id": normalized_id,
                    "rank": rank,
                    "product_id": evidence["product_id"],
                    "product_name": evidence["product_name"],
                    "category": evidence["category"],
                    "score": self.score(evidence),
                    "supporting_customers": supporters,
                }
            )
        return results

    def recommend_all(self, n: int = 3) -> list[dict[str, Any]]:
        """Recommend for every known customer, including empty-result placeholders."""
        self._validate_count(n, "n")
        rows: list[dict[str, Any]] = []
        for customer_id in sorted(self.customer_products):
            recommendations = self.recommend(customer_id, n)
            rows.extend(recommendations or [self._empty_recommendation(customer_id)])
        return rows

    @staticmethod
    def _empty_recommendation(customer_id: str) -> dict[str, Any]:
        return {
            "customer_id": customer_id,
            "rank": "",
            "product_id": "",
            "product_name": "",
            "category": "",
            "score": "",
            "supporting_customers": "",
        }

    def export_csv(
        self,
        rows: list[dict[str, Any]],
        path: str | Path = OUTPUT_PATH,
    ) -> None:
        """Write recommendations with one or more rows for every known customer."""
        output_rows = [dict(row) for row in rows]
        represented_customers: set[str] = set()
        for row in output_rows:
            customer_id = self._require_customer(row.get("customer_id", ""))
            row["customer_id"] = customer_id
            represented_customers.add(customer_id)

        for customer_id in sorted(set(self.customer_products) - represented_customers):
            output_rows.append(self._empty_recommendation(customer_id))

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {
                    column: row.get(column, "")
                    for column in self.CSV_COLUMNS
                }
                for row in output_rows
            )