export $(cat .env)
gsutil -m cp model/* gs://$MODEL_BUCKET/
gsutil -m cp data/*.csv gs://$DATA_BUCKET/

cp -r model ./monitoring/
cp -r data ./monitoring/
