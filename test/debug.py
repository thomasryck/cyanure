from cyanure.estimators import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale
import numpy as np


def test_LogisticRegression_elastic_net_objective(C, multiplier):
    # Check that training with a penalty matching the objective leads
    # to a lower objective.
    # Here we train a logistic regression with l2 (a) and elasticnet (b)
    # penalties, and compute the elasticnet objective. That of a should be
    # greater than that of b (both objectives are convex).
    n_samples=1000
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = scale(X)
    lambda_1 = 1.0 / C / n_samples

    lr_enet = LogisticRegression(
        penalty="elasticnet",
        solver="qning-miso",
        random_state=0,
        lambda_1=lambda_1 * multiplier,
        lambda_2=lambda_1 * (1 - multiplier),
        fit_intercept=False, verbose=True
    )
    print("This is elasticnet.")
    lr_enet.fit(X, y)

    lr_enet_intercept = LogisticRegression(
        penalty="elasticnet",
        solver="qning-miso",
        random_state=0,
        lambda_1=lambda_1 * multiplier,
        lambda_2=lambda_1 * (1 - multiplier),
        fit_intercept=True, verbose=True
    )
    print("This is elasticnet with intercept.")
    lr_enet_intercept.fit(X, y)

    lr_l1 = LogisticRegression(
        penalty="l1", solver="qning-miso", random_state=0, lambda_1=lambda_1, fit_intercept=False, verbose=True
    )
    print("This is l1.")
    lr_l1.fit(X, y)

    if True in np.isnan(lr_enet.coef_):
        print("Elasticnet")
        print(lambda_1)

    if True in np.isnan(lr_enet.coef_):
        print("Intercept")
        print(lambda_1)

    if True in np.isnan(lr_l1.coef_):
        print("L1")
        print(lambda_1)



for input_parameter in  np.logspace(-2, 2, 8):    
    for multiplier in [0.1, 0.5, 0.9] :
        test_LogisticRegression_elastic_net_objective(input_parameter, multiplier)