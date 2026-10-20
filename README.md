# next-best-action-recsys

Next-best-product recommendation for existing customers: given a customer's purchase history, rank the
products most likely to be in their next basket. Built on the real transaction log of a UK online
retailer (UCI Online Retail II: 802,632 purchase lines, 5,852 customers, 4,619 products, 36,594 baskets,
Dec 2009 to Dec 2011).

The project compares a Transformer recommender against strong classical baselines. It then combines
them with a learning-to-rank re-ranker and runs an offline A/B test the way a marketing team would
before a launch.

## Results (test: the next basket of 2,339 customers after 1 Sep 2011, mean 26 items)

| Model | Hit@10 | Recall@10 | NDCG@10 | New-item hit@10 | Catalogue coverage |
|---|---:|---:|---:|---:|---:|
| Popularity (last 30 days) | 39.3% | 3.4% | 0.077 | 22.2% | 0.2% |
| Repeat (time-decayed purchase history) | 68.9% | 13.6% | 0.249 | 1.1% | 54.2% |
| Item-KNN (co-purchase cosine) | 51.4% | 5.9% | 0.124 | 24.7% | 19.2% |
| Implicit ALS (matrix factorisation) | 55.5% | 8.9% | 0.135 | 24.4% | 31.8% |
| SASRec (item-level Transformer) | 28.9% | 2.1% | 0.061 | 9.3% | 3.8% |
| **Basket Transformer** (this project) | **75.0%** | 15.0% | 0.267 | 4.1% | 31.2% |
| **LightGBM LambdaRank re-ranker** over all of the above | 75.0% | **15.3%** | **0.276** | **12.4%** | 31.8% |

How to read the columns:
- **Hit@10:** the customer's next basket contains at least one of the 10 recommendations.
- **New-item hit@10:** the same, but counting only products the customer has never bought (discovery).

**Repeat purchases dominate this data.** "Show them what they bought before" beats every collaborative
filtering model, which is common for retail and grocery baskets. Any model worth shipping has to beat it.

**Item-level SASRec fails here.** It flattens baskets into sequences and learns to predict the next item
in an order that does not exist. Validation NDCG was 0.069 against 0.278 for Repeat.

**The basket Transformer is the fix** (`src/nba/basket_transformer.py`). How it works:
- Each step is a whole basket: its items are mean-pooled and the time since the previous basket is
  added.
- A causal Transformer reads the basket sequence and is trained with a listwise softmax over every item
  in the next basket.
- **The repeat term.** Each item's score also gets the customer's history with that item: how often
  they bought it, how recently, and whether it was in the latest basket. These features are scaled by a
  per-customer "repeat propensity" read from the Transformer state.

Without the repeat term it reached 0.12 validation NDCG. With it, 0.306, against 0.278 for Repeat. On
test it beats Repeat by 7% NDCG and 6 points of hit rate.

**The re-ranker adds discovery.** It is LightGBM LambdaRank, trained on the validation period with every
model's scores and ranks plus recency and popularity features. It roughly triples new-item hits
(4.1% to 12.4%) while keeping the Transformer's accuracy. Its most important features are the basket
Transformer's rank and score, then Repeat's rank.

## Offline A/B test (top 5 recommendations; conversion = next basket contains one of them)

Test customers are randomly split 50/50 between control and treatment (`src/nba/abtest.py`):

| Treatment vs control | Control | Treatment | Lift | p-value (two-proportion z) | Paired McNemar p | Customers per arm needed |
|---|---:|---:|---:|---:|---:|---:|
| Basket Transformer vs Repeat | 58.4% | 63.3% | +4.9 pts | 0.015 | 1e-14 | 766 |
| Re-ranker vs Repeat | 58.4% | 62.6% | +4.2 pts | 0.037 | 4e-12 | 1,028 |
| Re-ranker vs Basket Transformer | 66.4% | 62.6% | –3.8 pts | 0.060 | 0.22 | not significant |
| Re-ranker vs Popularity | 28.6% | 62.6% | +34 pts | <1e-10 | <1e-10 | 32 |

The table also reports a bootstrap confidence interval and the sample size needed per arm, at 80% power
and α = 0.05, to detect the observed lift.

The randomised split answers "would this win in a live test?". The paired McNemar test scores both
policies on every customer, which offline replay allows, and is far more sensitive.

Recommendation: ship the basket Transformer against Repeat. In a live test it needs about 770 customers
per arm. Use the re-ranker where discovery matters: it does not lose top-5 conversion significantly
(paired p = 0.22) and it finds three times as many new products.

## Data and evaluation

- **Cleaning** (`src/nba/data.py`) removes cancellations (invoices starting with C), returns,
  non-product lines (postage, fees, adjustments) and guest checkouts.
- **Temporal splits.** Validation uses history before 1 Jun 2011 to predict each customer's next
  basket after it; test uses history before 1 Sep 2011.
- **Warm start only.** Only customers with history are scored, since the task is recommending to
  existing customers.
- **Model selection.** Hyperparameters and the re-ranker are fitted on validation only; test is scored
  once.
- **Getting the data.** The data comes from UCI (CC BY 4.0). The `prepare-data` workflow downloads it
  and publishes Parquet to the `data` branch.

## Run it

```
pip install -e ".[test]"
mkdir -p data && git fetch origin data && git show origin/data:online_retail_ii.parquet > data/online_retail_ii.parquet
python scripts/run_experiment.py --data data/online_retail_ii.parquet --epochs 20
pytest -q
```

Results go to `docs/results/metrics.json` and `docs/results/ab_test.json`. The Transformers train on a
laptop CPU in about 20 to 40 minutes each.

## Limits

- **Replay is not a live test.** Recommendations can change what people buy, and replay cannot see
  that. The A/B numbers are the offline estimate a team would use to decide whether a live test is
  worth running.
- **One retailer, mostly wholesale buyers.** Baskets are large (26 items on average), so hit rates are
  higher than they would be for typical consumer baskets.
