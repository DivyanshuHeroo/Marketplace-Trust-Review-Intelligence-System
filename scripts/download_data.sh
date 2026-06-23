#!/usr/bin/env bash
# Download the 9 Olist e-commerce CSVs into data/raw/.
# Source: Hugging Face mirror of the public Kaggle "Brazilian E-Commerce Public
# Dataset by Olist" (CC BY-NC-SA 4.0).
#
# Usage:  bash scripts/download_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../data/raw"
BASE="https://huggingface.co/datasets/aviahYadler/Olist_Ecommerce_Dataset/resolve/main"

mkdir -p "$RAW_DIR"

FILES=(
  olist_customers_dataset.csv
  olist_geolocation_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_orders_dataset.csv
  olist_products_dataset.csv
  olist_sellers_dataset.csv
  product_category_name_translation.csv
)

echo "Downloading Olist dataset to $RAW_DIR ..."
for f in "${FILES[@]}"; do
  echo "  - $f"
  curl -s -L --max-time 180 -o "$RAW_DIR/$f" "$BASE/$f"
done

echo "Done. Files:"
ls -lh "$RAW_DIR"/*.csv
