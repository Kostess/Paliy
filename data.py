menu = [
    {"name": "Главная", "url": "/"},
    {"name": "KNN", "url": "p_knn"},
    {"name": "Линейная", "url": "p_linear"},
    {"name": "Логистическая", "url": "p_logistic"},
    {"name": "Дерево", "url": "p_three"},
    {"name": "Нейрон", "url": "p_neuro"},
    {"name": "Документация по API", "url": "doc_api"},
    {"name": "Распознавание одежды", "url": "p_fashion"},
    {"name": "Распознавание еды", "url": "p_food"},
    {"name": "Распознавание частей ПК", "url": "p_pc"},
    {"name": "Детектирование", "url": "p_detection"},
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

food_names = [
    "rice",
    "eels on rice",
    "pilaf",
    "chicken-'n'-egg on rice",
    "pork cutlet on rice",
    "beef curry",
    "sushi",
    "chicken rice",
    "fried rice",
    "tempura bowl",
    "bibimbap",
    "toast",
    "croissant",
    "roll bread",
    "raisin bread",
    "chip butty",
    "hamburger",
    "pizza",
    "sandwiches",
    "udon noodle",
    "tempura udon",
    "soba noodle",
    "ramen noodle",
    "beef noodle",
    "tensin noodle",
    "fried noodle",
    "spaghetti",
    "Japanese-style pancake",
    "takoyaki",
    "gratin",
    "sauteed vegetables",
    "croquette",
    "grilled eggplant",
    "sauteed spinach",
    "vegetable tempura",
    "miso soup",
    "potage",
    "sausage",
    "oden",
    "omelet",
    "ganmodoki",
    "jiaozi",
    "stew",
    "teriyaki grilled fish",
    "fried fish",
    "grilled salmon",
    "salmon meuniere",
    "sashimi",
    "grilled pacific saury",
    "sukiyaki",
    "sweet and sour pork",
    "lightly roasted fish",
    "steamed egg hotchpotch",
    "tempura",
    "fried chicken",
    "sirloin cutlet",
    "nanbanzuke",
    "boiled fish",
    "seasoned beef with potatoes",
    "hambarg steak",
    "beef steak",
    "dried fish",
    "ginger pork saute",
    "spicy chili-flavored tofu",
    "yakitori",
    "cabbage roll",
    "rolled omelet",
    "egg sunny-side up",
    "fermented soybeans",
    "cold tofu",
    "egg roll",
    "chilled noodle",
    "stir-fried beef and peppers",
    "simmered pork",
    "boiled chicken and vegetables",
    "sashimi bowl",
    "sushi bowl",
    "fish-shaped pancake with bean jam",
    "shrimp with chill source",
    "roast chicken",
    "steamed meat dumpling",
    "omelet with fried rice",
    "cutlet curry",
    "spaghetti meat sauce",
    "fried shrimp",
    "potato salad",
    "green salad",
    "macaroni salad",
    "Japanese tofu and vegetable chowder",
    "pork miso soup",
    "chinese soup",
    "beef bowl",
    "kinpira-style sauteed burdock",
    "rice ball",
    "pizza toast",
    "dipping noodles",
    "hot dog",
    "french fries",
    "mixed rice",
    "goya chanpuru"
]

pc_names = [
    "cables",
    "case",
    "cpu",
    "gpu",
    "hdd",
    "headset",
    "keyboard",
    "microphone",
    "monitor",
    "motherboard",
    "mouse",
    "ram",
    "speakers",
    "webcam"
]
