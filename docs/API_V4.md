# Version 4 portfolio decision contract

`/v4/portfolio/recommendations` serves the released TX_3 context actor. The bundled model contains **64 ordered SKU IDs**. Each request must provide either `item_ids` in that exact order or the `sku_order_sha256` returned by `/v4/health`; providing both checks both. This prevents a valid-looking but misordered vector from silently ordering for the wrong products. v1–v3 have their historical request contracts.

| Field | Contract |
| --- | --- |
| `day` | Nonnegative integer, zero-based M5 day index used for the weekly phase |
| `item_ids` | Optional only if `sku_order_sha256` is supplied; 64 ordered strings |
| `sku_order_sha256` | Optional only if `item_ids` is supplied; exact value from health metadata |
| `stock` | 64 nonnegative integer on-hand unit counts, in SKU order |
| `pipeline` | Three rows by 64 columns of nonnegative integer units arriving in one, two and three days |
| `last_sales` | 64 rows by seven columns of nonnegative integer sales, oldest to newest |

For example, this **shape-only demo** uses zero inventories and sales. Replace those arrays with an actual inventory snapshot before any meaningful use:

```python
import json
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8000"
with urlopen(base + "/v4/health") as response:
    metadata = json.load(response)
ids = metadata["item_ids"]
n = len(ids)
payload = {
    "day": 1800,
    "item_ids": ids,
    "stock": [0] * n,
    "pipeline": [[0] * n for _ in range(3)],
    "last_sales": [[0] * 7 for _ in range(n)],
}
request = Request(
    base + "/v4/portfolio/recommendations",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    decision = json.load(response)
print(decision["policy_type"], decision["spend"], decision["orders"])
```

The response includes the ordered SKU quantities, spend, selected policy type, decision day, model and manifest hashes, source-data hash, SKU-order hash and `sku_order_verified`. The health response exposes the active policy, promotion flag, SKU count and ordered IDs. Malformed state, mismatched SKU identity, an already over-capacity inventory position or invalid quantities yield HTTP 422. An unavailable or invalid model bundle yields HTTP 503. The allocator enforces shared purchase budget and storage capacity.

The service is an **offline research prototype**. Its hashes identify model and data versions for audit; they are not signatures or an authorization system. An enterprise deployment needs authenticated requests, live inventory reconciliation, a shadow run, monitoring and a new evaluation period before the policy can control purchasing.
