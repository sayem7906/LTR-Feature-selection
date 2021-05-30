import pandas as pd
import matplotlib.pyplot as plt
'''ohsumed'''
'''td2004'''
'''mq2008'''
'''mq2007'''


df =pd.read_csv('../Files/MQ2007/ndcg_result_MQ2007.csv')
#dataset = df.loc[df['Dataset'] == 'mushroom']
dataset = 'MQ2007'
score = "NDCG@10"
n = 'n2'
neighbor  = ', Neighbor: Insert'
algo = ' (SA)'
print(df)

# ay = df.plot.scatter(x="Feature_Number", y="Avg_NDCG@10")
# ay1 = df.plot.line(x="Feature_Number", y="Avg_NDCG@10",ax=ay)

# ax1 = df.plot.line(x="Feature_Number", y="Best_NDCG@10",ax=ay1)
# ax = df.plot.scatter(x="Feature_Number", y="Best_NDCG@10",ax=ax1)

plt.ylabel(score)
plt.xlabel('Number of features')
plt.ylim(0.275,.5)
# plt.xlim(1,46)


plt.errorbar(df['Feature_Number'], df[n+'s1_avg'],yerr=df[n+'s1_std'],label='Geometric scheme',ecolor='b',color='b')
plt.scatter(df['Feature_Number'], df[n+'s1_avg'],c='b')

# plt.plot(df['Feature_Number'], df[n+'s1_avg'])

plt.errorbar(df['Feature_Number'], df[n+'s2_avg'],yerr=df[n+'s2_std'],label='Log scheme',ecolor='r',color='r')
plt.scatter(df['Feature_Number'], df[n+'s2_avg'],c='r')
# plt.plot(df['Feature_Number'], df[n+'s2_avg'])

plt.errorbar(df['Feature_Number'], df[n+'s3_avg'],yerr=df[n+'s3_std'],label='Fast scheme',ecolor='g',color='g')
plt.scatter(df['Feature_Number'], df[n+'s3_avg'],c='g')

# plt.scatter(df['Feature_Number'], df['Best_NDCG@10'])
# plt.plot(df['Feature_Number'], df['Best_NDCG@10'])


plt.title(score+' score on '+dataset + neighbor + algo)
plt.legend(loc='lower right')
plt.show()

# ax = df.plot(x="clusters", y=["kmeans_precision %", "kmedoids_precision %"])
# ax.set_ylabel('Bcubed Precision %')
# plt.title('Bcubed Precision on '+dataset)
# plt.show()

# ax = df.plot(x="clusters", y=["kmeans_recall %", "kmedoids_recall %"])
# ax.set_ylabel('Bcubed Recall %')
# plt.title('Bcubed recall on '+dataset)
# plt.show()

