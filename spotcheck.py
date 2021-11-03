"""Functions to load datasets, define models, evaluate models on dataset and summarize results.
"""

import warnings
import calendar

from numpy import mean, std
from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier


def load_online_shoppers_dataset(data_frac=1.0, rs=None):
    """Load the dataset, returns X and y elements"""

    data = read_csv('data/online_shoppers_intention.csv')
    # Shuffle and sample data
    data = data.sample(frac=data_frac, random_state=rs)

    list_to_label_encode = ['VisitorType', 'Weekend', 'Revenue']
    le = LabelEncoder()
    for name in list_to_label_encode:
        data[name] = le.fit_transform(data[name].values)

    # Month name -> number
    d = dict((v, k) for k, v in enumerate(calendar.month_abbr))
    d['June'] = 6
    data.Month = data.Month.map(d)

    # Create dataset
    y = data.Revenue.values
    X = data.drop('Revenue', axis=1).values
    print(f"No of True labels = {sum(y == 1)} (Out of {y.shape[0]} examples)")
    return X, y


def load_htru2_dataset(data_frac=1.0, rs=None):
    """Load the pulsar dataset, returns X and y elements"""

    names = ['ip_mean', 'ip_std', 'ip_ek', 'ip_sk', 'dm-snr_mean', 'dm-snr_std', 'dm-snr_ek', 'dm-snr_sk', 'label']
    data = read_csv('data/HTRU_2.csv', sep=',', names=names)
    data = data.sample(frac=data_frac, random_state=rs)

    y = data['label'].values
    X = data.drop('label', axis=1).values
    print(f"No of True labels = {sum(y == 1)} (Out of {y.shape[0]} examples)")
    return X, y


def load_otto_dataset(data_frac=1.0, rs=None):
    """Load the otto group products dataset, returns X and y elements"""

    data = read_csv('data/otto_train.csv')
    data = data.sample(frac=data_frac, random_state=rs)
    le = LabelEncoder()
    data.target = le.fit_transform(data.target.values)

    array = data.values
    X = array[:, 1:94]
    y = array[:, 94]
    return X, y


def define_models(models=None):
    """Create a dict of standard models to evaluate {name:object}"""

    # linear models
    if models is None:
        models = dict()
    models['logistic'] = LogisticRegression(solver='lbfgs', max_iter=200, class_weight='balanced')
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['ridge-' + str(a)] = RidgeClassifier(alpha=a)
    models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
    models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    models['lda'] = LinearDiscriminantAnalysis()
    models['qda'] = QuadraticDiscriminantAnalysis(reg_param=0.1)

    # non-linear models
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models['knn-' + str(k)] = KNeighborsClassifier(n_neighbors=k)
    models['cart'] = DecisionTreeClassifier()
    models['extra'] = ExtraTreeClassifier()
    models['svml'] = SVC(kernel='linear', class_weight='balanced')
    models['svmp'] = SVC(kernel='poly', class_weight='balanced')
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models['svmr' + str(c)] = SVC(C=c, class_weight='balanced')
    models['bayes'] = GaussianNB()

    # ensemble models
    n_trees = 200
    models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
    models['bag'] = BaggingClassifier(n_estimators=n_trees)
    models['rf'] = RandomForestClassifier(n_estimators=n_trees)
    models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
    models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=0.1, max_depth=3, subsample=0.7)
    models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)

    print('Defined %d models' % len(models))
    return models


def define_gbm_models(models=None, use_xgb=True):
    """Define gradient boosting models"""

    if models is None:
        models = dict()
    # define config ranges
    rates = [0.01, 0.1]
    trees = [50, 100, 200]
    ss = [0.5, 0.7, 1.0]
    depth = [3, 5, 8]

    # add configurations
    for l in rates:
        for e in trees:
            for s in ss:
                for d in depth:
                    cfg = [l, e, s, d]
                    if use_xgb:
                        name = 'xgb-' + str(cfg)
                        models[name] = XGBClassifier(learning_rate=l, n_estimators=e, subsample=s, max_depth=d,
                                                     use_label_encoder=False, eval_metric='logloss')
                    else:
                        name = 'gbm-' + str(cfg)
                        models[name] = GradientBoostingClassifier(learning_rate=l, n_estimators=e, subsample=s,
                                                                  max_depth=d)
    print('Defined %d models' % len(models))
    return models


def make_pipeline(model):
    """Create a feature preparation pipeline for a model"""
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


def evaluate_model(X, y, model, folds, metrics):
    """Evaluate a single model"""
    # create the pipeline
    pipeline = make_pipeline(model)
    # evaluate model
    scores = cross_validate(pipeline, X, y, scoring=metrics, cv=folds, n_jobs=1)
    return scores


def robust_evaluate_model(X, y, model, folds, metric):
    """Evaluate a model and try to trap errors and and hide warnings"""
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            scores = evaluate_model(X, y, model, folds, metric)
    except:
        scores = None
    return scores


def evaluate_models(X, y, models, folds=5, metrics=None, multilabel=False):
    """evaluate a dict of models {name:object}, returns {name:score}"""

    if metrics is None:
        metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'] if multilabel \
            else ['accuracy', 'precision', 'recall', 'f1']
        print("Metrics:", metrics)

    results = dict()
    for name, model in models.items():
        # evaluate the model
        scores = robust_evaluate_model(X, y, model, folds, metrics)
        # scores = evaluate_model(X, y, model, folds, metrics)
        # show process
        if scores is not None:
            # store a result
            results[name] = scores['test_' + metrics[-1]]
            mean_score = [
                mean(scores['test_' + metrics[0]]), mean(scores['test_' + metrics[1]]),
                mean(scores['test_' + metrics[2]]), mean(scores['test_' + metrics[-1]])
            ]
            print('>%s: %.4f  %.4f  %.4f  %.4f' % (name, mean_score[0], mean_score[1], mean_score[2], mean_score[3]))
        else:
            print('>%s: error' % name)
    return results


def summarize_results(results, maximize=True, top_n=10):
    """print and plot the top n results"""

    # check for no results
    if len(results) == 0:
        print('no results')
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i + 1, name, mean_score, std_score))
    # boxplot for the top n
    pyplot.boxplot(scores, labels=names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    pyplot.savefig('spotcheck_classification.png')
