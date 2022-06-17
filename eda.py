import dataframe as d
import norm as n

db = d.df.describe()  # for dataframe
isnull = d.df.isnull().sum() # for dataframe

ndb = n.ndf.describe() # for normalize
nisnull = n.ndf.isnull().sum() # for normalize

nmmdb = n.nmmdf.describe() # for minmax
nmmisnull = n.nmmdf.isnull().sum() # for minmax
#ww = n.ndf.corr()