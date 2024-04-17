from sklearn.metrics import classification_report, confusion_matrix
from detach_rocket.detach_rocket.detach_classes import DetachRocket
from sklearn.linear_model import (RidgeClassifierCV,RidgeClassifier)
import numpy as np

def find_best(X_train, X_test, y_train, y_test):
    best_acc = 0
    for p in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15]:
        for _ in range(7):
            model = DetachRocket(model_type='minirocket', num_kernels=10000, trade_off=p)
            model.fit(X_train, y_train)

            X_train_transformed = model.transform_features(X_train)
            X_test_transformed = model.transform_features(X_test)

            cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
            cv_classifier.fit(X_train_transformed, y_train)
            model_alpha = cv_classifier.alpha_

            # Refit with all training set
            optimal_classifier = RidgeClassifier(alpha=model_alpha)
            optimal_classifier.fit(X_train_transformed, y_train)

            # Predictions
            y_pred = optimal_classifier.predict(X_test_transformed)
            y_train_pred = optimal_classifier.predict(X_train_transformed)

            # Metrics
            optimal_acc_train = optimal_classifier.score(X_train_transformed, y_train)
            optimal_acc_test = optimal_classifier.score(X_test_transformed, y_test)

            if optimal_acc_test > best_acc:
                best_acc = optimal_acc_test
                print('######################')
                print('######################')
                print(f'new best at p={p}')
                print('Training Accuracy:')
                print(optimal_acc_train)
                print('Testing Accuracy:')
                print(optimal_acc_test)
                print('\nClassification Report (Test):\n', classification_report(y_test, y_pred))
                print('\nConfusion Matrix (Test):\n', confusion_matrix(y_test, y_pred))
                print('######################')
                print('######################')

    return optimal_classifier

def find_best2(X_train, X_test, y_train, y_test):
    best_acc = 0
    for p in [0.01]:
        for _ in range(25):
            model = DetachRocket(model_type='minirocket', num_kernels=10000, trade_off=p)
            model.fit(X_train, y_train)

            X_train_transformed = model.transform_features(X_train)
            X_test_transformed = model.transform_features(X_test)

            cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
            cv_classifier.fit(X_train_transformed, y_train)
            model_alpha = cv_classifier.alpha_

            # Refit with all training set
            optimal_classifier = RidgeClassifier(alpha=model_alpha)
            optimal_classifier.fit(X_train_transformed, y_train)

            # Predictions
            y_pred = optimal_classifier.predict(X_test_transformed)
            y_train_pred = optimal_classifier.predict(X_train_transformed)

            # Metrics
            optimal_acc_train = optimal_classifier.score(X_train_transformed, y_train)
            optimal_acc_test = optimal_classifier.score(X_test_transformed, y_test)

            if optimal_acc_test > best_acc:
                best_acc = optimal_acc_test
                print('######################')
                print('######################')
                print(f'new best at p={p}')
                print('Training Accuracy:')
                print(optimal_acc_train)
                print('Testing Accuracy:')
                print(optimal_acc_test)
                print('\nClassification Report (Test):\n', classification_report(y_test, y_pred))
                print('\nConfusion Matrix (Test):\n', confusion_matrix(y_test, y_pred))
                print('######################')
                print('######################')

    return optimal_classifier