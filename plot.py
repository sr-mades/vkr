#import dataframe as d
import os
import seaborn as sb
import matplotlib.pyplot as plt

# def learn():
#     plt.title('обучение')
#     plt.xlabel('epochs')
#     plt.ylabel('mse')
#     plt.plot()
#
#     return

def heatcorr(df):
    sb.heatmap(data=df, annot=True)
    plt.show()
    return

def pltshow(df, nm):
    os.mkdir('/Volumes/SRV/project/vkr/fig/{0}/'.format(nm))
    sb.kdeplot(data=df, shade=True)
    plt.savefig('/Volumes/SRV/project/vkr/fig/{0}/{1}_kde.png'.format(nm, nm))
    plt.show()
    sb.pairplot(data=df)
    plt.savefig('/Volumes/SRV/project/vkr/fig/{0}/{1}_pairplot.png'.format(nm, nm))
    plt.show()
    j = 0
    for i in df.columns:
        j += 1
        sb.kdeplot(data=df[i], shade=True)
      #  plt.show()
        plt.savefig('/Volumes/SRV/project/vkr/fig/{0}/{column}.png'.format(nm, column=j))
        plt.clf()
        sb.boxplot(x=df[i])
       # plt.show()
        plt.savefig('/Volumes/SRV/project/vkr/fig/{0}/{column}_1.png'.format(nm, column=j))
        plt.clf()

    return
