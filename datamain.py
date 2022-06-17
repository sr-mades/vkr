import pandas as pd
import dataframe as d
import eda
import plot
import norm as n
import Linear as l
import deeplearn as dl


# with pd.ExcelWriter("/Volumes/SRV/project/vkr/df.xlsx") as writer:
#     d.df.to_excel(writer, sheet_name='dataframe')
#     eda.db.to_excel(writer, sheet_name='describe')
#     eda.isnull.to_excel(writer, sheet_name='isnull')
#     n.ndf.to_excel(writer, sheet_name='norm')
#     eda.ndb.to_excel(writer, sheet_name='norm describe')
#     eda.nisnull.to_excel(writer, sheet_name='norm isnull')
#     n.ndf.corr().to_excel(writer, sheet_name='norm corr')
#     n.nmmdf.to_excel(writer, sheet_name='minmax')
#     eda.nmmdb.to_excel(writer, sheet_name='minmax describe')
#     eda.nmmisnull.to_excel(writer, sheet_name='minmax isnull')
#     n.nmmdf.corr().to_excel(writer, sheet_name='minmax corr')

#plot.pltshow(d.df, 'dataframe')
#plot.pltshow(n.ndf, 'norm')
#plot.pltshow(n.nmmdf, 'normmm')
#plot.heatcorr(n.ndf.corr())

#lcoef, linter = l.linearreg(n.ndf,'Модуль упругости при растяжении, ГПа')
#lcoef2, linter2 = l.linearreg(n.ndf,'Прочность при растяжении, МПа')
#

dl.tensorf(n.ndf, 'Соотношение матрица-наполнитель')


