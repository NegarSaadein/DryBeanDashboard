import pandas as pd
from ucimlrepo import fetch_ucirepo

# Load Dry Bean dataset
dry_bean = fetch_ucirepo(id=602)

X = dry_bean.data.features.copy()
y = dry_bean.data.targets.squeeze()

df = pd.concat([X, y.rename("Class")], axis=1)
