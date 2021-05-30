import pandas as pd
import matplotlib.pyplot as plt


df =pd.read_csv('Local_beam_search_result_neighbor#1_beam_width#7_dataset#5.csv')
#dataset = df.loc[df['Dataset'] == 'mushroom']
dataset = 'MSLR-10K'
# print(df)

# ay = df.plot.scatter(x="Feature_Number", y="Avg_NDCG@10")
# ay1 = df.plot.line(x="Feature_Number", y="Avg_NDCG@10",ax=ay)

# ax1 = df.plot.line(x="Feature_Number", y="Best_NDCG@10",ax=ay1)
# ax = df.plot.scatter(x="Feature_Number", y="Best_NDCG@10",ax=ax1)
"""Feature_Number	Avg_NDCG@10	Std_NDCG@10	Best_NDCG@10	Avg_MAP@10	        Std_MAP@10	Best_MAP@10
"""
plt.ylabel('MAP@10')
plt.xlabel('Number of features')
plt.ylim(0.12,.18)
# plt.xlim(1,46)


plt.errorbar(df['Feature_Number'], df['Avg_MAP@10'],yerr=df['Std_MAP@10'],ecolor='r',color='b')
plt.scatter(df['Feature_Number'], df['Avg_MAP@10'],c='b')

# plt.plot(df['Feature_Number'], df['n1s1_avg'])

# plt.errorbar(df['Feature_Number'], df['n1s2_avg'],yerr=df['n1s2_std'],label='Log scheme',ecolor='r',color='r')
# plt.scatter(df['Feature_Number'], df['n1s2_avg'],c='r')
# # plt.plot(df['Feature_Number'], df['n1s2_avg'])

# plt.errorbar(df['Feature_Number'], df['n1s3_avg'],yerr=df['n1s3_std'],label='Fast scheme',ecolor='g',color='g')
# plt.scatter(df['Feature_Number'], df['n1s3_avg'],c='g')

# plt.scatter(df['Feature_Number'], df['Best_NDCG@10'])
# plt.plot(df['Feature_Number'], df['Best_NDCG@10'])


plt.title('MAP@10 score on '+dataset+'(Neighbor: Swap) - Local beam')
plt.legend()
plt.show()

# ax = df.plot(x="clusters", y=["kmeans_precision %", "kmedoids_precision %"])
# ax.set_ylabel('Bcubed Precision %')
# plt.title('Bcubed Precision on '+dataset)
# plt.show()

# ax = df.plot(x="clusters", y=["kmeans_recall %", "kmedoids_recall %"])
# ax.set_ylabel('Bcubed Recall %')
# plt.title('Bcubed recall on '+dataset)
# plt.show()

