# Confusion matrix values for training data
TN_train, FP_train, FN_train, TP_train = 378, 26, 7, 389

# Accuracy
accuracy_train = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
# Precision
precision_train = TP_train / (TP_train + FP_train)
# Recall
recall_train = TP_train / (TP_train + FN_train)
# F1 Score
f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

roc_auc_train = TP_train / (FP_train + TN_train)

print("Training Data Metrics:")
print(f"Accuracy: {accuracy_train * 100:.2f}%")
print(f"Precision: {precision_train * 100:.2f}%")
print(f"Recall: {recall_train * 100:.2f}%")
print(f"F1 Score: {f1_train * 100:.2f}%")
print(f"Roc Auc Score {roc_auc_train}")
# Confusion matrix values for validation data
TN_val, FP_val, FN_val, TP_val = 81, 15, 9, 95

# Accuracy
accuracy_val = (TP_val + TN_val) / (TP_val + TN_val + FP_val + FN_val)
# Precision
precision_val = TP_val / (TP_val + FP_val)
# Recall
recall_val = TP_val / (TP_val + FN_val)
# F1 Score
f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

print("Validation Data Metrics:")
print(f"Accuracy: {accuracy_val * 100:.2f}%")
print(f"Precision: {precision_val * 100:.2f}%")
print(f"Recall: {recall_val * 100:.2f}%")
print(f"F1 Score: {f1_val * 100:.2f}%")
