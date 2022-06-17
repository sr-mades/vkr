import pandas as pd

xbp = pd.read_excel('/Users/user/PycharmProjects/VKR/data/X_bp.xlsx')
xnup = pd.read_excel('/Users/user/PycharmProjects/VKR/data/X_nup.xlsx')
xbp.drop('Unnamed: 0', axis=1, inplace=True) # удаляем
xnup.drop('Unnamed: 0', axis=1, inplace=True) # удаляем
df = pd.concat([xbp, xnup], axis=1, join='inner') # объединение