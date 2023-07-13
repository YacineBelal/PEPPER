import seaborn as sns 
import csv 
import pandas as pd 
import matplotlib.pyplot as plt 

# sender, receiver, none normalized weight, normalized weight, round
to_dump = [10,'no', 74, 0.4, 0.5, 55,"Pepper"]

def to_csv_file():
    with open("list_weights_given.csv","a") as output:
        writer = csv.writer(output, delimiter=",")
        writer.writerow(to_dump)

# to_csv_file()

df = pd.read_csv("list_weights_given.csv")
# df.to_csv("list_weights_given.csv", header=["Sender", "IsAttacker", "Receiver", "None-normalized-weights", "Normalized-weights", "Round","Setting"], index=False)


# df = df[df["Round"] > 500]

sns.catplot(data = df, x="Setting", y="Normalized-weights", hue="IsAttacker", kind="box").set(title="normalized weights over the whole training")
plt.show()

# sns.catplot(data = df, x= pd.cut(df[df['Setting'] == 'Pepper']["Round"],[0,200,400,600]), y="Normalized-weights", hue="IsAttacker", kind="box")


