#!/usr/bin/env python
# coding: utf-8

# # Analyze Results from AUT LLM Study
# 
# <a href="https://colab.research.google.com/github/massivetexts/llm_aut_study/blob/main/notebooks/Analyze_AUT_Runs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This code analyzes runs evaluating LLM embedding models, WEM semantic models, and LLM fine-tuned models over a large, nine-source dataset of Alternate Uses Tasks data.
# 
# It is the analysis of experimental results from Organisciak, P., Acar, S., Dumas, D., & Berthiaume, K. (2022). Beyond Semantic Distance: Automated Scoring of Divergent Thinking Greatly Improves with Large Language Models. http://dx.doi.org/10.13140/RG.2.2.32393.31840.
# 

# In[1]:


from pathlib import Path
import pandas as pd
import os
from scipy.stats import pearsonr, kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#@title Load Data. 
#@markdown Select `load_from_online` if downloading results from Organisciak et al (2022).
load_from_online = True #@param {type: 'boolean'}
if load_from_online:
    # !wget -O evaluation.zip https://github.com/massivetexts/llm_aut_study/blob/main/results/evaluation.zip?raw=true
    # !unzip -o evaluation.zip
    evaldir = Path('evaluation')
else:
    # path to evaluation files
    evaldir = Path('drive/MyDrive/Grants/MOTES/Data/evaluation')


# In[15]:

script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)
evaldir = Path(os.path.join(root_dir, 'evaluation'))

#@title Process Data
#@markdown Run this cell to run data preprocessing.
paths = list(evaldir.glob('*/*csv'))
dfs = []
for path in paths:
    if path.parent.stem not in ['gt_main2', 'all', 'gt_byprompt', 'fewshot']:
        continue
    df = pd.read_csv(path, index_col=0)
    df = df[[col for col in ['id', 'model', 'participant', 'prompt', 'target', 'predicted', 'src', 'proportion', 'nexamples', 'ncompletions', 'total_tokens'] if col in df.columns]]
    df['split'] = path.parent.stem
    dfs.append(df)
df = pd.concat(dfs)
if 'proportion' not in df.columns:
    df['proportion'] = 1
else:
    df.proportion = df.proportion.fillna(1)
def normalize_model_name(x):
    x = x.lower()
    x = x.replace('_', '-')
    x = x.replace('-text-similarity', '')
    x = x.replace('gpt-', 'gpt3_')
    x = x.replace('-001', '')
    return x
df.model = df.model.apply(normalize_model_name)
df.sample()


# In[16]:


# additional versions of baselines that I crunched, but are excessive to include
drop_models = ['semdis-tasa-nf-m', 'semdis-cbowbncwikiukwac-nf-m', 'semdis-cbowsubtitle-nf-m', 
               'semdis-cbowukwacsubtitle-nf-m', 
               'semdis-glove-nf-m', 'ocs-stop', 'ocs-weight', 'ocs-target']

model_order = ['semdis-mean', 'semdis-tasa-nf-m', 'semdis-cbowbncwikiukwac-nf-m', 
               'semdis-cbowsubtitle-nf-m', 'semdis-cbowukwacsubtitle-nf-m', 
               'semdis-glove-nf-m', 'ocs-main', 'ocs-weight', 'ocs-stop',
               'ocs-target', 'st5-base', 'st5-large', 'st5-3b', 'gpt3_emb-ada',
               'gpt3_emb-babbage', 't5-base', 'gpt3-finetune-ada', 'gpt3-ada',
               'gpt3-finetune-babbage', 'gpt3-babbage',
               'gpt3-curie', 'gpt3-davinci']
remaining = [x for x in set(df.model.unique()).difference(model_order)]
model_order += remaining


# ## Performance on main dataset

# In[17]:


split = 'gt_main2'
print(split.upper().center(69,'-'))
x = df[(df.split == split) & (df.proportion == 1)]
all = x.groupby('model').corr().loc[(slice(None), 'target'), 'predicted'].reset_index()[['model','predicted']].rename(columns={'predicted': 'ALL'}).set_index('model')
y = x.groupby(['model', 'src']).corr().loc[(slice(None), slice(None), 'target'), 'predicted'].reset_index()
z = y.pivot(index='model', columns='src', values='predicted')
all = all.merge(z, left_index=True, right_index=True)
all = all.loc[[m for m in model_order if m in all.index]]

all = all.reset_index()
all[['class', 'subclass']] = all.model.str.split('-', 1, expand=True).values
def categorize(model):
    model = model.lower()
    if model in ['semdis', 'ocs']:
        return 'baseline'
    elif model in ['st5', 'gpt3_emb']:
        return 'LLM embed'
    elif model in ['t5', 'gpt3']:
        return 'LLM fine-tune'
    else:
        'unknown'
all['category'] = all['class'].apply(categorize, 1)

def make_pretty(styler, include_gradient=True, gradient_range=(0,1), precision=3):
    styler.set_caption("Performance")
    #styler.format_index(lambda v: v.strftime("%A"))
    styler.format(precision=precision)
    if include_gradient:
        styler.background_gradient(axis=None, vmin=gradient_range[0], vmax=gradient_range[1], cmap="YlGnBu")
    styler.highlight_max(props='font-weight:bold;', axis = 0)
    return styler
forpaper = all[~all.model.isin(drop_models)].set_index(['category', 'model']).drop(columns=['class','subclass'])
forpaper.style.pipe(lambda x: make_pretty(x, gradient_range=(forpaper.min().min(), forpaper.max().max())))


# In[18]:


forpaper = all[~all.model.isin(drop_models)].groupby('category').mean().loc[['baseline', 'LLM embed', 'LLM fine-tune']]
forpaper.style.pipe(lambda x: make_pretty(x, gradient_range=(forpaper.min().min(), forpaper.max().max()), precision=3))


# In[19]:


#@title Do the models over or under-estimate the results?
#@markdown Look at distribution. For example, the ocs-main distribution and skew:
x[x.model=='ocs-main'].target.plot(kind='kde')
print("Skew:", x[x.model=='ocs-main'].target.skew())


# In[20]:


#@markdown Skew for all models
x[~x.model.isin(drop_models)].groupby('model').predicted.skew().sort_values()


# In[21]:


#@markdown Plot distributions
x = df[(df.split == split) & (df.proportion == 1) & ~df.model.isin(drop_models)]
rows = 3
cols = len(x.model.unique())//rows
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10,5))
fig.tight_layout(h_pad=2)
gt = x[x.model=='ocs-main'].target # the *target* distribution, not predicted
print('gt skew', gt.skew().round(2))
for i, model in enumerate([m for m in model_order if m in x.model.unique()]):
    ax = axs[(i)//cols, (i)%cols]
    z = x[x.model == model]
    ksstat = kstest(gt, z.predicted) #Kolmogorov-Smirnov Goodness-of-Fit Test
    gt.plot(kind='kde', ax=ax, color='green', linestyle='--')
    z.predicted.plot(kind='kde', ax=ax, title=model)
    ax.label_outer()
    ax.set_xticks([])
    ax.set_xlabel(f'skew={z.predicted.skew().round(2)}, D={np.round(ksstat.statistic, 2)}')
#fig.delaxes(axs[1, cols-1])


# In[23]:


def get_sig(x):
    return pearsonr(x['target'], x['predicted'])[1]

no_na = x[~x[['target', 'predicted']].isna().any(1)]
allp = no_na.groupby(['model']).apply(get_sig).sort_values().reset_index().rename(columns={0: 'ALL'}).set_index('model')
yp = no_na.groupby(['model', 'src']).apply(get_sig).reset_index()
zp = yp.pivot(index='model', columns='src', values=0)
allp = allp.merge(zp, left_index=True, right_index=True)
allp = allp.loc[[m for m in model_order if m in allp.index and m not in drop_models]]
print("Statistical significance, marked only for non-significant results")
allp.round(2).replace(0, '')


# In[24]:


#@title By Prompt
#@markdown Which prompts are easier/tougher to score?
x = df[(df.split == 'gt_main2') & (df.proportion == 1)].copy()
#print('temporary - fixing cat names, but should re-run eval and remove this code at the end'.upper())
#x.prompt = x.prompt.str.replace('hat cap', 'hat').str.replace('lead pencil', 'pencil').str.replace('light bulb', 'lightbulb').str.replace('lightbulbs', 'lightbulb').str.replace('spoons', 'spoon').str.replace('soccer ball', 'ball')
#all = x.groupby('model').corr().loc[(slice(None), 'target'), 'predicted'].reset_index()[['model','predicted']].rename(columns={'predicted': 'ALL'}).set_index('model')
y = x.groupby(['model', 'prompt']).corr().loc[(slice(None), slice(None), 'target'), 'predicted'].reset_index()
z = y.pivot(index='model', columns='prompt', values='predicted')
#all = all.merge(z, left_index=True, right_index=True).round(3)
z['M'] = z.mean(1)
z['SD'] = z.std(1)
z = z.round(3).loc[[col for col in model_order if col in z.index and col not in drop_models]]
#display(z)
#plt.figure(figsize=(15,8))
#ax = sns.heatmap(z, annot=True, vmin=0, cbar=False, cmap="Blues")
#ax.xaxis.tick_top()
z = z.reset_index()
z[['class', 'subclass']] = z.model.str.split('-', 1, expand=True).values
z['category'] = z['class'].apply(categorize, 1)
forpaper = z[~z.model.isin(drop_models)].set_index(['category', 'model']).drop(columns=['class','subclass'])
forpaper.style.pipe(lambda x: make_pretty(x, gradient_range=(forpaper.min().min(), forpaper.max().max()), precision=2))


# In[25]:


forpaper = z[~z.model.isin(drop_models)].groupby('category').mean().loc[['baseline', 'LLM embed', 'LLM fine-tune']]
forpaper.style.pipe(lambda x: make_pretty(x, gradient_range=(forpaper.min().min(), forpaper.max().max()), precision=2))


# In[26]:


x[['class', 'subclass']] = x.model.str.split('-', 1, expand=True).values
x['category'] = x['class'].apply(categorize, 1)
avg_ft = x.groupby(['category', 'model', 'prompt'])[['target','predicted']].corr()[['predicted']].iloc[::2].reset_index().groupby(['category', 'prompt'])[['predicted']].mean().loc['LLM fine-tune']
# entropy calculated when processing dataset
entropy = pd.Series({'backpack': 3.481, 'ball': 5.937, 'book': 6.135, 'bottle': 6.679, 'box': 7.901, 'brick': 8.483, 'fork': 5.946, 'hat': 5.914, 'knife': 7.627, 'lightbulb': 5.921, 'pants': 6.031, 'paperclip': 7.189, 'pencil': 5.915, 'rope': 7.574, 'shoe': 5.875, 'shovel': 5.759, 'sock': 5.916, 'spoon': 5.929, 'table': 6.051, 'tire': 5.959, 'toothbrush': 5.907}, name='entropy')
y = avg_ft.merge(entropy, left_index=True, right_index=True).sort_values('entropy') #.plot(kind='scatter', x='predicted', y='entropy')
print("Correlation of entropy with performance")
y.corr().round(3).values[0,1]


# # Performance with different training sizes

# In[27]:


split = 'gt_main2'
totaln = 16081
print(f'Assuming n={totaln} (hardcoded)')
targetmodels = df[df.proportion < 1].model.unique().tolist()
x = df[(df.split == split) & (df.model.isin(targetmodels))]
all = x.groupby(['model', 'proportion']).corr().loc[(slice(None), slice(None), 'target'), 'predicted']
all = all.reset_index()[['model','proportion', 'predicted']].rename(columns={'predicted': 'corr'})
ax = None
for model in all[(all.proportion > 0.01) & (all.proportion < 1)].model.unique():
    ax = all[(all.model == model) & (all.proportion > 0.01)].plot(x='proportion', y='corr', ax=ax, label=model, style='.-')

# add top axis
def prop2n(prop):
    return totaln * prop

def n2prop(n):
    return n / totaln

ax.set_ylabel('Performance (correlation)')
ax.set_xlabel('proportion of all training labels')
secax = ax.secondary_xaxis('top', functions=(prop2n, n2prop))
secax.set_xlabel('# of training labels')
all['n'] = all.proportion.apply(prop2n, 0).astype(int)
all[all.proportion >= 0.01].pivot(index=['proportion', 'n'], columns='model', values='corr').dropna(axis=1)


# In[28]:


x = all.pivot(index=['proportion', 'n'], columns='model', values='corr')
x['gpt3-ada'] / x['gpt3-babbage']


# # Unseen Prompts
# 
# Look by Prompt.

# # Few Shot

# In[29]:


split = 'fewshot'
print(split.upper().center(69,'-'))
x = df[(df.split == split) & (df.proportion == 1)]
groupvars = ['nexamples', 'ncompletions']
all = x.groupby(groupvars).corr().loc[(slice(None), slice(None), 'target'), 'predicted'].reset_index()[groupvars+['predicted']].rename(columns={'predicted': 'ALL'}).set_index(groupvars)
y = x.groupby(groupvars+['prompt']).corr().loc[(slice(None), slice(None), slice(None), 'target'), 'predicted'].reset_index()
z = y.pivot(index=groupvars, columns='prompt', values='predicted')
all = all.merge(z, left_index=True, right_index=True).round(3)
#all = all.loc[[m for m in model_order if m in all.index]]
display(all)
plt.show(sns.heatmap(all, annot=True, vmin=0, cbar=False, cmap="Blues"))


# # Costs for GPT-3

# In[30]:


x = df[~df.total_tokens.isna() & (df.split == 'gt_main2') & (df.proportion == 1)]
costs = x.groupby('model')['total_tokens'].aggregate(['count', 'sum', 'mean'])
costs['price/1k'] = [0.0016, 0.0024, 0.012, 0.12]
costs['avg cost/item'] = costs['mean']/1000 * costs['price/1k']
costs


# In[31]:


costs['total_cost'] = (costs['sum']/1000 * costs['price/1k']).round(2)
costs['responses/dollar'] = (1/costs['avg cost/item']).round(0).astype(int)
costs


# In[32]:


print(costs[['responses/dollar']].to_markdown())


# In[33]:


x = df[df.split == 'gt_byprompt'].copy()
all = x.groupby('model').corr().loc[(slice(None), 'target'), 'predicted'].reset_index()[['model','predicted']].rename(columns={'predicted': 'ALL'}).set_index('model')
y = x.groupby(['model', 'prompt']).corr().loc[(slice(None), slice(None), 'target'), 'predicted'].reset_index()
z = y.pivot(index='model', columns='prompt', values='predicted')
z['M'] = z.mean(1)
z['SD'] = z.std(1)
z = z.round(3)
all = all.merge(z, left_index=True, right_index=True).round(3).loc[[model for model in model_order if model in all.index]]
display(all[~all.index.isin(drop_models)])
plt.figure(figsize=(15,8))
ax = sns.heatmap(all[~all.index.isin(drop_models)], annot=True, vmin=0, cbar=False, cmap="Blues")
ax.xaxis.tick_top()


# # Look at responses
# 
# Look at responses. This requires loading the ground truth.
# 
# This is most some curiosity-related stuff - inspecting qualitatively what the responses look like where the machine and judges agree.

# In[36]:


base_dir = Path('drive/MyDrive/Grants/MOTES/Data/aut_ground_truth')
data_subdir = "gt_main2"
get_ipython().system('cp {base_dir}/{data_subdir}.tar.gz .')
get_ipython().system('rm -rf data')
get_ipython().system('tar -xf {data_subdir}.tar.gz')
data_dir = Path(f"data/{data_subdir}")
print("Data decompressed to", data_dir)

# load test data
testdata = pd.DataFrame([pd.read_json(x, orient='index')[0] for x in (data_dir / 'test').glob('*json')])
testdata.sample()


# Look at distribution of difference between model and human judges.

# In[46]:


y = df[(df.split=='gt_main2') & (df.model=='gpt3-davinci')]
z = y.merge(testdata, on=['id'])
z['diff'] = z.target_x - z.predicted
z['diff'].plot(kind='hist')


# Find some examples at different levels of creativity, where the model matched GT.

# In[ ]:


z[(z['diff'] == 0) & (z['prompt_x'] == 'pants')].sort_values('predicted', ascending=False)[['prompt_x', 'response', 'predicted']].head(20)

