# import pandas as pd
# data=pd.read_csv("dataset.csv")

# data=data.drop(columns=['id','category','severity','language','source'])
# data.rename(columns={'binary_label':'label'}, inplace=True)
# data.to_csv("dataset_.csv",index=False)

import pandas as pd
df = pd.read_csv('cyberbullying_tweets.csv')

df.rename(columns={'tweet_text': 'text','cyberbullying_type':'label'}, inplace=True)
df['label'] = df['label'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

df['text']=df['text'].str.replace(r"[^a-zA-Z0-9 ]", '', regex=True)
df = df.dropna(subset=['text'])
df=df[df['text']!='']
df.to_csv('dataset.csv', index=False)