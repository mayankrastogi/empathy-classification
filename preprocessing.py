import numpy
import pandas
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Borrowed from sveitser's answer to a question at Stack Overflow:
        https://stackoverflow.com/a/25562948/4463881

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self._fill = pandas.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )

        return self

    def transform(self, X, y=None):
        return X.fillna(self._fill)


class DataFrameOneHotEncoder(TransformerMixin):
    def __init__(self):
        """
        Generate dummy variables for categorical features

        Based on Zygmunt Z's blog post at:
        http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/

        """

    def fit(self, X, y=None):
        df = pandas.get_dummies(X)
        self._columns = df.columns
        return self

    def transform(self, X, y=None):
        d = pandas.get_dummies(X)
        self._fix_columns(d, self._columns)
        return d

    def _add_missing_dummy_columns(self, d, columns):
        missing_cols = set(columns) - set(d.columns)
        for c in missing_cols:
            d[c] = 0

    def _fix_columns(self, d, columns):

        self._add_missing_dummy_columns(d, columns)

        # make sure we have all the columns we need
        assert (set(columns) - set(d.columns) == set())

        extra_cols = set(d.columns) - set(columns)
        if extra_cols:
            print("extra columns found:", extra_cols)

        d = d[columns]
        return d


def binarize(y, threshold=0, force=False):
    # Binarize only if y is not already binarized;
    # e.g. while testing, the test set will be already binarized
    if force or numpy.max(y) > 1:
        return pandas.Series(numpy.where(y > threshold, 1, 0), index=y.index)
    else:
        return y