# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import matplotlib.pyplot as plt
import re


# %%
episode = pd.read_csv('input/Episodes.csv')
desc = pd.read_csv('input/Description.csv')


# %%
episode[episode['episode_id']=="E69"]["release_date"]

# %% [markdown]
# - If you have listened to that episode, Sanyanam had said he has turned into age of 23. So his DOB is: 27/05/1997

# %%
episode['heroes'].isna().sum()

# %% [markdown]
# - 11 episodes where Sanyanam Bhutani himself was the hero
# (it includes 1 trailer, AMA, 9 episodes of fastai review classes)

# %%
episode.head()


# %%
f = open('input/Raw Subtitles/E1.txt')


# %%
def extract_transcript(fn, save=False, save_path=''):
    "Takes transcript and converts it to `DataFrame`"
    pat = r'([A-Za-z]|\s+)\s([0-9]{0,2}:{0,1}[0-9]{1,2}:[0-9][0-9])'
    f = open(fn, "r")
    t = True
    df = pd.DataFrame(columns = ['Time', 'Speaker', 'Text'])
    i = 0
    first = True
    while t:
        line = f.readline()
        if line == '': t = False
        i += 1
        line = re.split(pat, line[:-1])
        if len(line) == 4:
            is_new = 1
            speak = line[0]
            time = line[2]
        while is_new == 1:
            if first:
                line = f.readline()
                for i in range(6):
                    l_c = f.readline()
                    if speak not in l_c and time not in l_c:
                        line += l_c
                i += 1
                first = False
            else:
                line = f.readline()
                i += 1
            if len(line) > 2 and line != '\n':
                line = line[:-1]
                df.loc[i] = [time, speak, line]
                df.reset_index()
            else:
                is_new = 0
    df.reset_index(drop=True, inplace=True)
    df['Text'] = df['Text'].replace('\n', '')
    if save:
        df.to_csv(save_path+fn.name[:-3] + 'csv', index=False, sep='|')
    return df


# %%
df = extract_transcript('input/Raw Subtitles/E1.txt')


# %%
df.head()


# %%
df.shape


# %%
df[df['Speaker']=='Sanyam Bhutani'].shape


# %%


