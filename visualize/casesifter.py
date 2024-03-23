import matplotlib.pyplot as plt 
import seaborn as sns
# color: https://www.biomooc.com/color/seabornColors.html

sns.set(font_scale=1.2)
sns.set_style('white')

plot_data = {
    'a': [3000, 6000, 9000, 1e10, 3000, 6000, 9000, 1e10, 3000, 6000, 9000, 1e10],
    'micro-F1(%)': [69.4, 69.9, 69.9, 69.4, 70.8, 71.2, 71.1, 71.1, 71.0, 71.0, 71.0, 70.9],
    'k': ['k=3', 'k=3', 'k=3', 'k=3', 'k=5', 'k=5', 'k=5', 'k=5', 'k=7', 'k=7', 'k=7', 'k=7']
}

for i in range(len(plot_data['a'])):
    if plot_data['a'][i] == 3000:
        plot_data['a'][i] = 1
    if plot_data['a'][i] == 6000:
        plot_data['a'][i] = 2    
    if plot_data['a'][i] == 9000:
        plot_data['a'][i] = 3
    if plot_data['a'][i] == 1e10:
        plot_data['a'][i] = 4


fig = plt.figure()
sns.lineplot(x='a', y='micro-F1(%)', data=plot_data, style='k', markers='o', dashes=False, hue='k', linewidth=3)

fig.axes[0].set_ylim(68.5, 72.5)
fig.axes[0].text(x=2-0.2, y=71.2 + 0.2, s='(2, 71.2)')
fig.axes[0].set(xlabel='α')
fig.axes[0].set_xticks(range(1,5))
fig.axes[0].set_xticklabels(['1', '2', '3', '1e10'])

# plt.title('test')
plt.show()





# fig = sns.lineplot(x='Lambda', y='micro-F1(%)', data=plot_data, style='l', markers='o', dashes=False)