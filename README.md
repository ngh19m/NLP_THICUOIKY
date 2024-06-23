              precision    recall  f1-score   support

       B-art       0.85      0.50      0.63        94
       B-eve       0.97      0.81      0.88        70
       B-geo       0.91      0.95      0.93      7558
       B-gpe       0.98      0.94      0.96      3142
       B-nat       0.71      0.50      0.59        40
       B-org       0.93      0.80      0.86      4151
       B-per       0.95      0.90      0.93      3400
       B-tim       0.98      0.93      0.95      4077
       I-art       0.96      0.83      0.89        84
       I-eve       0.97      0.94      0.95        65
       I-geo       0.98      0.98      0.98      1462
       I-gpe       0.89      0.73      0.80        33
       I-nat       0.90      0.69      0.78        13
       I-org       0.98      0.98      0.98      3394
       I-per       0.97      0.99      0.98      3406
       I-tim       0.95      0.95      0.95      1251
           O       0.99      1.00      1.00    177590

    accuracy                           0.99    209830
   macro avg       0.93      0.85      0.88    209830
weighted avg       0.99      0.99      0.99    209830

Accuracy: 0.9867
Precision (macro): 0.9334
Recall (macro): 0.8482
F1-score (macro): 0.8847
F1-score theo entity: 0.9087


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_test, y_test):
    """
    Đánh giá mô hình bằng accuracy, precision, recall, F1-score,
    và F1-score theo entity.
    """
    y_pred = model.predict(X_test)

    # Chuyển đổi y_test và y_pred thành dạng danh sách phẳng
    y_test_flat = [tag for sublist in y_test for tag in sublist]
    y_pred_flat = [tag for sublist in y_pred for tag in sublist]

    # In báo cáo chi tiết
    print(classification_report(y_test_flat, y_pred_flat))

    # Tính toán và in các chỉ số chung
    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    precision = precision_score(y_test_flat, y_pred_flat, average='macro')
    recall = recall_score(y_test_flat, y_pred_flat, average='macro')
    f1 = f1_score(y_test_flat, y_pred_flat, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    # Tính toán F1-score theo entity
    entity_f1 = f_measure(y_test, y_pred)
    print(f"F1-score theo entity: {entity_f1:.4f}")
