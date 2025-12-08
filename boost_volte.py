from app import vn

print("Boosting knowledge for 'tbl_volte_service'...")

# 1. Add Documentation
# Explicitly link the Chinese term "VoLTE业务" to the table name
doc_text = "The table `db_provision`.`tbl_volte_service` stores all information about VoLTE services (VoLTE业务). Use this table when querying for VoLTE status, users, or configurations."
vn.train(documentation=doc_text)

print(f"Added documentation: {doc_text}")

# 2. Add a sample SQL pair (Optional but recommended)
# This teaches the model exactly how to query it
sample_question = "查询所有开通了 VoLTE 业务的用户"
sample_sql = "SELECT * FROM `db_provision`.`tbl_volte_service` WHERE status = 'ACTIVE'" # Assuming 'status' exists, adjust if needed
vn.train(question=sample_question, sql=sample_sql)

print(f"Added sample SQL pair: {sample_question} -> {sample_sql}")

print("\nDone! The AI should now be able to find this table.")
