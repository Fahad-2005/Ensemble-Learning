
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dataset_generator as dg
from bagging import BaggingClassifierScratch
from random_forest import RandomForestClassifierScratch
from adaboost import AdaBoostClassifierScratch

def run_bagging_experiment():
    print("\nExecuting Bagging Experiments...")
    X, y = dg.generate_bagging_datasets()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)
    
    estimator_range = [1, 5, 15]
    train_scores, test_scores = [], []
    
    for n_est in estimator_range:
        bag = BaggingClassifierScratch(n_estimators=n_est, max_depth=6)
        bag.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, bag.predict(X_train)))
        test_scores.append(accuracy_score(y_test, bag.predict(X_test)))
        
    plt.figure(figsize=(6, 4))
    plt.plot(estimator_range, train_scores, label="Train Accuracy", marker='o')
    plt.plot(estimator_range, test_scores, label="Validation Accuracy", marker='s')
    plt.title("Bagging Ensemble Scale Scaling Impact")
    plt.xlabel("Number of Base Trees")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("bagging_evaluation.png")
    plt.close()
    print(f"Bagging execution complete. Peak Validation Accuracy: {max(test_scores)*100:.2f}%")

def run_rf_experiment():
    print("\nExecuting Random Forest Experiments...")
    X, y = dg.generate_class_imbalance_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)
    
    rf = RandomForestClassifierScratch(n_estimators=10, max_depth=8)
    oob_score = rf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Random Forest Manual OOB Accuracy Score: {oob_score*100:.2f}%")
    print(f"Random Forest Validation Score on Imbalanced Data: {test_acc*100:.2f}%")

def run_adaboost_experiment():
    print("\nExecuting AdaBoost Sequential Noise Variance Experiments...")
    X, y_5, y_15, y_30 = dg.generate_adaboost_noise_datasets()
    
    noise_scenarios = {"5% Noise": y_5, "15% Noise": y_15, "30% Noise": y_30}
    accuracies = []
    
    for name, y_vec in noise_scenarios.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_vec, test_size=0.3, random_state=47)
        ada = AdaBoostClassifierScratch(n_estimators=15)
        ada.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, ada.predict(X_te))
        accuracies.append(acc)
        print(f"AdaBoost Model Performance under {name}: {acc*100:.2f}%")
        
    plt.figure(figsize=(6, 4))
    plt.bar(noise_scenarios.keys(), accuracies, color=['green', 'orange', 'red'])
    plt.title("AdaBoost Robustness Profile Across Noise Scales")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig("adaboost_noise_degradation.png")
    plt.close()

if __name__ == "__main__":
    run_bagging_experiment()
    run_rf_experiment()
    run_adaboost_experiment()
    print("\nExecution complete. All artifact charts generated for report compilation.")