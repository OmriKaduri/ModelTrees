import pandas as pd


def machine_dataset():
    data = pd.read_csv('machine.data',
                       names=['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'])
    labels = data['ERP']
    samples = data.drop(columns=['ERP'])
    return samples, labels


def forestfires_dataset():
    data = pd.read_csv('forestfires.csv')
    labels = data['area']
    samples = data.drop(columns=['area'])
    return samples, labels


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    d = d[columns]
    for a_col, b_col in zip(d.columns, columns):
        assert(a_col == b_col)
    return d


def DatasetFactory(dataset_name):
    if dataset_name == 'machine': return machine_dataset()
    if dataset_name == 'forestfires': return forestfires_dataset()
    print("Can't find that dataset. DUDE.")
