source .venv/bin/activate

# if fashion dataset, run this
if [ "$DATASET" = "fashion" ]; then
python main.py \
  --image /enviorment/store.jpg \
  --docs_json  data/fashion/attributes \
  --crops_dir  data/fashion/segments/crops \
  --out_links  data/fashion/links/segment_links.json \
  --out_meta  data/fashion/attributes/ground_truth_mapping.json \
  --query "${QUERY_ID:-c6_c_6.json}"
fi

# if redial dataset, run this
if [ "$DATASET" = "movie" ]; then
python  main.py \
  --image  enviorment/store.jpg \
  --docs_json  data/movie/attributes \
  --crops_dir  data/movie/segments/crops \
  --out_links  data/movie/links/segment_links.json \
  --out_meta  data/movie/attributes/ground_truth_mapping.json \
  --query "${QUERY_ID:-c6_c_6.json}"
fi

# if cosrec dataset, run this
if [ "$DATASET" = "retail" ]; then
python  main.py \
  --image  enviorment/store.jpg \
  --docs_json  data/retail/attributes \
  --crops_dir  data/retail/segments/crops \
  --out_links  data/retail/links/segment_links.json \
  --out_meta  data/retail/attributes/ground_truth_mapping.json \
  --query "${QUERY_ID:-c6_c_6.json}"
fi
