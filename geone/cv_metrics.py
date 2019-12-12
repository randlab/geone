import numpy as np

def brier_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    return np.mean([2*p[np.where(estimator.classes_ == i)] - np.sum(p**2) - 1 for p, i in zip(y_pred_proba, y_true)])

def zero_one_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    return np.mean([zero_one_rule(p, i, estimator.classes_) for p, i in zip(y_pred_proba, y_true)])

def linear_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    linear_scores = [linear_rule(p, i, estimator.classes_) for p, i in zip(y_pred_proba, y_true)]
    return np.mean(linear_scores)

def balanced_brier_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    brier_scores = [2*p[np.where(estimator.classes_ == i)] - np.sum(p**2) - 1 for p, i in zip(y_pred_proba, y_true)]
    return _balance_score(y_true, brier_scores, estimator.classes_)

def balanced_zero_one_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    zero_one_scores = [zero_one_rule(p, i, estimator.classes_) for p, i in zip(y_pred_proba, y_true)]
    return _balance_score(y_true, zero_one_scores, estimator.classes_)

def balanced_linear_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    linear_scores = [linear_rule(p, i, estimator.classes_) for p, i in zip(y_pred_proba, y_true)]
    return _balance_score(y_true, linear_scores, estimator.classes_)

def zero_one_rule(y_pred_proba, y_true, classes):
    max_proba = np.max(y_pred_proba)
    modes = classes[y_pred_proba == (np.ones_like(y_pred_proba)*max_proba)]
    if y_true in modes:
        score = 1/len(modes)
    else:
        score = 0
    return score

def linear_rule(y_pred_proba, y_true, classes):
    return y_pred_proba[np.where(y_true==classes)]

def _generate_prediction(estimator, X):
    if hasattr(estimator, 'previous_X_') and np.all(estimator.previous_X_ == X):
        y_pred_proba = estimator.previous_y_
    else:
        y_pred_proba = estimator.predict_proba(X)
    return y_pred_proba

def _balance_score(y_true, scores, classes):
    cum_score = np.zeros_like(classes, dtype=float)
    n_elements_in_class = np.zeros_like(classes, dtype=float)

    for y, score in zip(y_true, scores):
        # classes encode the categories, retrieve index of the true value
        index = np.where(classes == y)
        cum_score[index] += score
        n_elements_in_class[index] += 1

    return np.mean(cum_score[np.nonzero(n_elements_in_class)]/n_elements_in_class[np.nonzero(n_elements_in_class)])
