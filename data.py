import pandas as pd
data=pd.read_csv("dataset.csv")

data=data.drop(columns=['id','category','severity','language','source'])
data.rename(columns={'binary_label':'label'}, inplace=True)
data.to_csv("dataset_.csv",index=False)