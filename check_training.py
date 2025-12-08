from app import vn
import pandas as pd

# Check if the table exists in the training data
print("Checking training data for 'tbl_volte_service'...")
training_data = vn.get_training_data()

# Filter for the table name
found = False
if training_data is not None and not training_data.empty:
    # Check in 'content' column which usually holds the DDL or documentation
    matches = training_data[training_data['content'].str.contains('tbl_volte_service', case=False, na=False)]
    if not matches.empty:
        print(f"FOUND! Found {len(matches)} records related to 'tbl_volte_service'.")
        print(matches[['id', 'training_data_type', 'content']].head())
        found = True
    else:
        print("NOT FOUND. 'tbl_volte_service' is not in the vector database.")
else:
    print("Vector database is empty.")

if not found:
    print("\n--- Attempting to retrain specifically for this table ---")
    # You can uncomment the following lines to force retrain if you want
    # vn.train(ddl="CREATE TABLE `db_provision`.`tbl_volte_service` (...)")
    pass
