import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme(style="whitegrid")


# color: https://www.biomooc.com/color/seabornColors.html



plot_data = {
    'Split': ['Valid', 'Test', 'Valid', 'Test'],
    'Micro-F1 (%)': [79.8, 79.6, 73.7, 67.7],
    'Split Methods': ['Random', 'Random', 'Chonological', 'Chonological']
}


# fig, ax = plt.subplots(1,2)


# sns.lineplot(x='Lambda', y='micro-F1(%)', data=plot_data, color=sns.xkcd_rgb['windows blue'], ax=ax[0])
# ax[0].set_ylim(68, 72)
# sns.set_color_codes("pastel")
fig = sns.barplot(data=plot_data, x='Split Methods', y='Micro-F1 (%)', hue='Split', width=0.6, palette='Paired')
fig.set_ylim(0, 100)
for i in fig.containers:
    fig.bar_label(i,)
# plt.title('test')
fig.set_xticklabels(['Random Split', 'Chonological Split'])
# fig.legend(['Random', 'Chonological'], title='Split Methods')

plt.ylabel('Micro-F1 (%)')
plt.show()


# plt.savefig('heatmap.pdf')
# from matplotlib.backends.backend_pdf import PdfPages
# pdf = PdfPages('foo.pdf')
# pdf.savefig(fig, height=10, width=18, dpi=500, bbox_inches='tight', pad_inches=0.5)
# plt.close()