# 读取comparison_results.csv person,adv1_orig2_similarity,adv2_orig1_similarity
# 计算adv1_orig2_similarity 大于0.3的数量
# 计算adv2_orig1_similarity 大于0.3的数量

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# 读取数据
df = pd.read_csv('comparison_results.csv')
df.head()
# df2 = pd.read_csv('comparison_results.csv')

# df 为合并取min的结果
# df = pd.concat([df, df2]).groupby('person').min().reset_index()
# df.head()

# %%
# 计算adv1_orig2_similarity 大于0.3的数量
adv1_orig2_count = (df['adv1_orig2_similarity'] > 0.3).sum()
adv1_orig2_count

# %%
# 计算adv2_orig1_similarity 大于0.3的数量
adv2_orig1_count = (df['adv2_orig1_similarity'] > 0.3).sum()
adv2_orig1_count
# %%
