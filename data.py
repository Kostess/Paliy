menu = [
    {"name": "Главная", "url": "/"},
    {"name": "KNN", "url": "p_knn"},
    {"name": "Линейная", "url": "p_linear"},
    {"name": "Логистическая", "url": "p_logistic"},
    {"name": "Дерево", "url": "p_three"},
    {"name": "Нейрон", "url": "p_neuro"},
    {"name": "Распознавание одежды", "url": "p_fashion"},
    {"name": "Документация по API", "url": "doc_api"},
]

metrics_linear_data = [
    {"name": "MSE", "value": "1.3509243001909827e-29"},
    {"name": "RMSE", "value": "6.754621500954914e-30"},
    {"name": "MSPE", "value": "5.403697200763931e-29"},
    {"name": "MAPE", "value": "8.500000000000002"},
    {"name": "SMAPE", "value": "1.831867990631502e-15"},
    {"name": "MASE", "value": "7.327471962526033e-15"},
    {"name": "MRE", "value": "inf"},
    {"name": "RMSLE", "value": "2.9624445261614147e-15"},
    {"name": "R-квадрат", "value": "1.0"},
    {"name": "Скорректированный R-квадрат", "value": "0.7986743400332134"},
]

metrics_classification_data = [
    {"name": "Confusion matrix", "type": "logistic", "value": {"Predicted": [73, 15], "Actual": [10, 82]}},
    {"name": "Confusion matrix", "type": "KNN", "value": {"Predicted": [72, 16], "Actual": [14, 78]}},
    {"name": "Confusion matrix", "type": "three", "value": {"Predicted": [70, 18], "Actual": [21, 71]}},
    {"name": "Accuracy", "type": "logistic", "value": 0.8611111111111112},
    {"name": "Accuracy", "type": "KNN", "value": 0.8333333333333334},
    {"name": "Accuracy", "type": "three", "value": 0.7833333333333333},
    {"name": "Precision", "type": "logistic", "value": 0.845360824742268},
    {"name": "Precision", "type": "KNN", "value": 0.8297872340425532},
    {"name": "Precision", "type": "three", "value": 0.797752808988764},
    {"name": "Recall", "type": "logistic", "value": 0.8913043478260869},
    {"name": "Recall", "type": "KNN", "value": 0.8478260869565217},
    {"name": "Recall", "type": "three", "value": 0.7717391304347826},
]