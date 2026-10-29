# Mama Mboga Online: Explainable Recommendations

This project recommends products using customers' purchase histories. It uses
only Python's standard library and can be run from the project directory:

```powershell
python main.py
```

By default, the program reads `data/purchases.csv` and writes
`output/recommendations.csv`. Paths and recommendation count can be changed:

```powershell
python main.py --input data/purchases.csv --output output/recommendations.csv --count 3
```

The CSV needs `customer_id`, `product_id`, and `product_name` columns.
`category` is optional and is left blank when absent. `quantity` is optional;
when present, it must be a positive whole number. `read_purchases(path)` returns
`(valid_rows, rejected_rows)`. Each rejected row includes a
`_rejection_reason`; missing required headers raise `ValueError`.

## Data-cleaning rules

- Customer and product IDs are stripped of surrounding whitespace and
  case-folded. For example, `" C001 "` and `"c001"` refer to the same ID.
- Rows with blank customer IDs, product IDs, or product names are rejected.
- If a quantity column exists, blank, non-integer, zero, and negative quantities
  are rejected. If there is no quantity column, quantity defaults to `1`.
- A row with extra values beyond the header is rejected as malformed; those
  values remain available under `_extra_values` on the rejected row.
- Repeated transaction rows are retained as valid input rows. Histories are sets,
  so a customer's repeated purchase of a product counts once. Popularity is the
  number of distinct customers who bought a product, so repeats do not inflate
  its popularity.
- Product name and category are stored once per product using the first
  non-empty values encountered.

## How recommendations work

1. `RecommendationEngine` builds customer-to-product sets, product metadata,
   and distinct-customer product popularity at initialization. Other methods
   use these indexes rather than rescanning the input rows.
2. Similarity is Jaccard similarity:
   `|products_a ∩ products_b| / |products_a ∪ products_b|`. Two empty sets have
   similarity `1.0`; one empty and one non-empty set have similarity `0.0`.
3. The top `k` neighbours are sorted by descending similarity, then by
   normalized customer ID. The customer is excluded, and neighbours below
   `min_similarity` are excluded. The default threshold is `0.0`.
4. Candidate products are products bought by at least one selected neighbour
   that the customer has not bought. If there are no selected neighbours, the
   fallback is the globally popular unseen catalog, ordered by the same scoring
   rule. This gives a known customer a useful suggestion even when the
   similarity threshold leaves no neighbours.
5. Scores combine three explainable signals:

   ```text
   0.7 × mean similarity of supporting neighbours
   + 0.2 × (supporting neighbours / selected neighbours)
   + 0.1 × (product popularity / maximum product popularity)
   ```

   The similarity term gives more weight to stronger neighbours. The support
   term rewards agreement among neighbours, and the popularity term gently
   favors broadly purchased items. Scores are deterministic; remaining ties are
   broken by support count, popularity, then product ID.

### Worked score example

Suppose customer `c1` bought maize. Neighbour `c2` bought maize and beans, and
`c3` bought maize and rice. Each neighbour has Jaccard similarity `1/2` to `c1`.
For beans, one of the two neighbours supports it, and suppose beans' distinct
customer popularity is `1` while the most popular product was bought by `4`
customers:

```text
0.7 × 0.5 + 0.2 × (1/2) + 0.1 × (1/4) = 0.475
```

Rice gets the same score; the product-ID tie-break decides which appears first.

Customers must have at least one valid purchase to be known to the engine.
Unknown customer IDs raise `KeyError`. A customer with no unseen products gets
no product recommendation. `recommend_all()` and the CSV exporter still include
that customer through a row with blank recommendation fields.

## Output

`recommendations.csv` has these columns:

```text
customer_id,rank,product_id,product_name,category,score,supporting_customers
```

The `supporting_customers` value lists neighbour IDs and their similarities.
Each known customer appears at least once, even if they have no recommendation.
The run prints valid and rejected row counts, a similarity matrix for the first
five customers, and a summary of generated recommendations.

## Tests

Run the standard-library test suite from the project directory:

```powershell
python -m unittest discover -v
```
