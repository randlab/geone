import numpy as np

def brier_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    return np.mean([2*p[np.where(estimator.classes_ == i)] - np.sum(p**2) - 1 for p, i in zip(y_pred_proba, y_true)])

def zero_one_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    return np.mean([zero_one_rule(p, i, estimator.classes_) for p, i in zip(y_pred_proba, y_true)])

def zero_one_rule(y_pred_proba, y_true, classes):
    max_proba = np.max(y_pred_proba)
    modes = classes[y_pred_proba == (np.ones_like(y_pred_proba)*max_proba)]
    if y_true in modes:
        score = 1/len(modes)
    else:
        score = 0
    return score


def balanced_brier_score(estimator, X, y_true):
    y_pred_proba = _generate_prediction(estimator, X)
    brier_scores = [2*prob[np.where(estimator.classes_ == i)] - np.sum(prob**2) - 1 for prob, i in zip(y_pred_proba, y_true)]
    
    cum_score = np.zeros_like(estimator.classes_, dtype=float)
    number_of_elements_in_class = np.zeros_like(estimator.classes_, dtype=float)

    for y, brier in zip(y_true, brier_scores):
        # classes encode the categories, retrieve index of the true value
        index = np.where(estimator.classes_ == y)
        cum_score[index] += brier
        number_of_elements_in_class[index] += 1

    return np.mean(cum_score[np.nonzero(number_of_elements_in_class)]/number_of_elements_in_class[np.nonzero(number_of_elements_in_class)])

def _generate_prediction(estimator, X):
    if hasattr(estimator, 'previous_X_') and np.all(estimator.previous_X_ == X):
        y_pred_proba = estimator.previous_y_
    else:
        y_pred_proba = estimator.predict_proba(X)
    return y_pred_proba
