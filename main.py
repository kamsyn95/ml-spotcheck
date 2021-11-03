"""Script to test different ML algorithms on 3 datasets:
- HTRU2,
- Online Shoppers Intention,
- Otto Group Products
"""

from argparse import ArgumentParser
from spotcheck import load_online_shoppers_dataset, load_htru2_dataset, load_otto_dataset
from spotcheck import define_models, define_gbm_models, evaluate_models, summarize_results


# Create parser of input arguments
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['htru2', 'shoppers', 'otto'], default='htru2')
parser.add_argument('--data_frac', type=float, default=0.1)
parser.add_argument('--gbm_models', action='store_true', default=False)


if __name__ == '__main__':
    args = parser.parse_args()

    # load dataset
    if args.dataset == 'shoppers':
        X, y = load_online_shoppers_dataset(data_frac=args.data_frac, rs=1)
    elif args.dataset == 'otto':
        X, y = load_otto_dataset(data_frac=args.data_frac, rs=1)
    else:
        X, y = load_htru2_dataset(data_frac=args.data_frac, rs=1)

    # get model list
    models = define_models()
    # add gradient boosting models
    if args.gbm_models:
        models = define_gbm_models(models)
    # evaluate models
    multilabel = True if args.dataset == 'otto' else False
    results = evaluate_models(X, y, models, multilabel=multilabel)
    # summarize results
    summarize_results(results)
