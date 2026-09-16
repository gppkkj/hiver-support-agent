"""
Filters the full Customer Support on Twitter export (twcs.csv, ~2.8M rows)
down to one brand's threads, and reconstructs (customer_text, brand_reply)
resolution pairs for the playbook.

Usage:
    python3 scripts/filter_brand.py --brand AppleSupport \
        --raw /mnt/user-data/uploads/twcs.csv --outdir data/

Produces, under --outdir:
    brand_raw.csv          all tweets belonging to the brand's threads
                            (brand's own tweets + the customer tweets they
                            replied to / were replied to by)
    resolved_pairs.csv     (customer_text, brand_reply) pairs mined from
                            real inbound->first outbound reply links
    first_contact.csv      first-inbound-message-of-thread only, i.e. the
                            population the golden set is sampled from
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Filter twcs.csv to one brand and mine support-resolution pairs."
    )
    ap.add_argument("--brand", required=True, help="Brand author_id, e.g. AppleSupport")
    ap.add_argument("--raw", required=True, help="Path to the Kaggle twcs.csv export")
    ap.add_argument("--outdir", default="data/", help="Directory for generated CSVs")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    outdir = Path(args.outdir)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(raw_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read raw dataset {raw_path}: {exc}") from exc
    required = {
        "tweet_id", "created_at", "in_response_to_tweet_id",
        "author_id", "inbound", "text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError(f"Raw dataset is empty: {raw_path}")
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    by_id = df.set_index("tweet_id", drop=False)

    brand_out = df[df["author_id"] == args.brand].copy()
    print(f"{args.brand}: {len(brand_out)} outbound tweets")

    # Resolution pairs: for each brand reply, the customer tweet it replied to,
    # restricted to cases where that customer tweet really is inbound (real
    # customer -> real brand reply, not brand -> brand or duplicate hops).
    pairs = []
    for _, row in brand_out.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if parent_id is None or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate tweet_id edge case
            parent = parent.iloc[0]
        if not bool(parent["inbound"]):
            continue
        pairs.append({
            "customer_tweet_id": parent["tweet_id"],
            "customer_text": parent["text"],
            "brand_reply": row["text"],
        })

    resolved = pd.DataFrame(pairs).drop_duplicates(subset=["customer_tweet_id"])
    print(f"Resolved pairs mined: {len(resolved)}")

    # First-contact population: inbound tweets @-mentioning the brand that
    # are themselves NOT a reply to anything (start of a thread) -- this is
    # what the golden set + live agent should be triaging, not follow-ups.
    mentions = df[
        df["inbound"]
        & df["text"].str.contains(f"@{args.brand}", case=False, na=False)
    ].copy()
    first_contact = mentions[mentions["in_response_to_tweet_id"].isna()]
    print(f"First-contact candidate population: {len(first_contact)}")

    resolved.to_csv(outdir / "resolved_pairs.csv", index=False)
    first_contact[["tweet_id", "created_at", "text"]].to_csv(
        outdir / "first_contact.csv", index=False
    )
    print(f"[3/3] Wrote resolved_pairs.csv and first_contact.csv to {outdir}")


if __name__ == "__main__":
    main()
