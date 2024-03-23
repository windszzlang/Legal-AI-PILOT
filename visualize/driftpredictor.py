import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

# color: https://www.biomooc.com/color/seabornColors.html

sns.set(font_scale=1.2)
sns.set_style('white')

# , 0.3
# , 69.7
# , 70.6
plot_data = {
    'Lambda(%)': [0, 5, 10, 15, 20],
    'micro-F1(%)': [70.6, 70.7, 71.2, 70.4, 70.3],
    'base': [70.6, 70.6, 70.6, 70.6, 70.6],
}


# fig, ax = plt.subplots(1,2)


# sns.lineplot(x='Lambda', y='micro-F1(%)', data=plot_data, color=sns.xkcd_rgb['windows blue'], ax=ax[0])
# ax[0].set_ylim(68, 72)
fig = sns.lineplot(x='Lambda(%)', y='micro-F1(%)', data=plot_data, linewidth=3, color=sns.xkcd_rgb['windows blue'], marker='o')
fig = sns.lineplot(x='Lambda(%)', y='base', data=plot_data, linestyle='dashed', color=sns.xkcd_rgb['windows blue'], linewidth=3)
fig.set_ylim(68, 72)
plt.xticks(np.arange(0, 21, 5))
# plt.yticks(np.arange(0, 20, 5))
fig.text(x=10-2, y=71.2 + 0.2, s='(10, 71.2)')
# plt.title('test')
plt.show()






# fig = sns.lineplot(x='Lambda', y='micro-F1(%)', data=plot_data, style='l', markers='o', dashes=False)