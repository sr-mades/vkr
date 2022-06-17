import dataframe as d
import sklearn.preprocessing as sp
import pandas as pd
import numpy as np

norm = sp.normalize(d.df, axis=0)
ndf = pd.DataFrame(norm, columns=d.df.columns)

nmm = sp.MinMaxScaler().fit_transform(ndf)
nmmdf = pd.DataFrame(nmm, columns=d.df.columns)
