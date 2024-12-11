import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Specify the path to your Excel file
file_path = '.xlsx'  

df = pd.read_excel(file_path)


modalities = ['CT', 'MR']


model1_results_all = df.iloc[:, 10]  
model2_results_all = df.iloc[:, 31]  

figure_list = [modalities, anatomical_sites]

for figure in figure_list:
    for content in figure:
        if content in modalities:
            filtered_indices = df[df.iloc[:, 4] == content].index

        model1_results = model1_results_all.loc[filtered_indices].dropna()
        model2_results = model2_results_all.loc[filtered_indices].dropna()

        if len(model1_results) > 1 and len(model2_results) > 1:
            t_stat, p_value = ttest_ind(model1_results, model2_results)

            plt.rc('font', size=20)  
            plt.figure(figsize=(8, 6))
            bplot = plt.boxplot(
                [model1_results, model2_results],
                labels=['LLM-RadSum', 'GPT4o'],
                patch_artist=True,  # Enables colored boxes
                boxprops=dict(color='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                showfliers=False,  
                capprops=dict(color='black'),
            )

            colors = ['#F09BA0', '#9BBBE1']  
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            plt.legend(
                [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors],
                ['LLM-RadSum', 'GPT4o'],
                loc='upper right',
                fontsize=18
            )

            y_max = max(model1_results.max(), model2_results.max())  
            y_annotation = y_max + 0.05 

            if 0.001 <= p_value < 0.05:
                plt.text(1.5, y_annotation, '*', ha='center', fontsize=22, color='black')
            elif p_value < 0.001:
                plt.text(1.5, y_annotation, '**', ha='center', fontsize=22, color='black')


            plt.title(f'{content}')
            plt.ylabel('Values')
            plt.ylim(top=y_annotation + 0.1)  
            plt.show()

            print(f"Content: {content}")
            print(f"T-test result: t-statistic = {t_stat:.3f}, p-value = {p_value:.3e}")
        else:
            print(f"Not enough data for content: {content}")

