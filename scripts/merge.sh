head -n 1 data/fetched.csv > data/merged.csv && tail -n +2 -q data/fetched.csv data/preprocessed.csv  >> data/merged.csv