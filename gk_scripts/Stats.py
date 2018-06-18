"""
    Collection of functions for statistical computation
"""

"""
    Takes frequency of positive class and shows breakdown of random prediction

    Rationale: make a tree map to find tp, tn, etc.
    input: 
        - freq: frequency of positive class
        - print_results: whether to print metrics
    output:
        - positive class precision, etc.
"""
def random_prediction(freq, print_results=True):
    tp = freq * freq
    tn = (1-freq) * (1-freq)
    fp = (1-freq) * (freq)
    fn = freq * (1-freq)

    pos_prec = tp / (tp + fp) 
    pos_rec = tp / (tp + fn)
    neg_prec = tn / (tn + fn)
    neg_rec = tn / (tn + fp)

    if print_results:
        print('AVG METRICS:\n')
        print('(+) Class Precision: {}\n'.format(round(pos_prec, 3)))
        print('(+) Class Recall: {}\n'.format(round(pos_rec, 3)))
        print('(-) Class Precision: {}\n'.format(round(neg_prec, 3)))
        print('(-) Class Recall: {}\n'.format(round(neg_rec, 3)))

