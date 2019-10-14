import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from pathlib import Path

def gridSearch(clf, params, scoring, X_train, y_train, X_test, y_test, refit=None, filename="randomForest", cv=5, random=False, n_iter=10):

    filepath = Path("./models")

    print("# Tuning hyper-parameters for %s" % refit)
    print()

    if random: # perform random search instead of exhaustive grid search
        print("Random ON")
        model = RandomizedSearchCV(clf, params, refit=refit, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        model = GridSearchCV(clf, params, refit=refit, cv=cv, scoring=scoring, n_jobs=-1)

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    results = model.cv_results_
    means_list = []
    stds_list = []
    for scorer in scoring:
        means_list.append(results['mean_test_%s' % scorer])
        stds_list.append(results['std_test_%s' % scorer])

    for i, means in enumerate(means_list):
        print("Scorer: %s\n" % scoring[i])
        for mean, std, params in zip(means, stds_list[i], model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 1.96, params))
        print()

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    newFilename = filename + "_{}.pkl".format(refit)
    print("Saving best model to file: %s" % str(filepath / newFilename))
    joblib.dump(model.best_estimator_, open(filepath / newFilename, 'wb'))