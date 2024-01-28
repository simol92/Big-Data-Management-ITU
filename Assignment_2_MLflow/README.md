You can test the model on my VM with: 

curl http://20.251.152.0:5000/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "columns": ["Speed", "Direction"],
    "data": [
      [12.96416, 4],
      [13.85824, 4],
      [15.19936, 4]
    ]
  }
}'
