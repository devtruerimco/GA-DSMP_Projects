# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data=pd.read_csv(path)
data.hist(column="Rating")


data=data[data["Rating"]<=5]
data.hist(column="Rating")


#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
#total_null=total_null.values.tolist()

percent_null=total_null/data.isnull().count()

missing_data=pd.concat([total_null,percent_null],axis=1,keys=["Total","Percent"])
print("missing_data:",missing_data)

data.dropna(inplace=True)
total_null_1=data.isnull().sum()
percent_null_1=total_null_1/data.isnull().count()

missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=["Total","Percent"])
print("missing_data_1:",missing_data_1)



# code ends here


# --------------

#Code starts here

g=sns.catplot(x="Category",y="Rating",kind="box",height=10,data=data)
g.set_xticklabels(rotation=90)
g.set_titles("Rating vs Category [BoxPlot]")
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print("value counts of Installs column(before rmoving +,):",data["Installs"].value_counts())

data['Installs'] = data['Installs'].str.replace("[+,]","").astype(int)
print("value counts of Installs column(after rmoving +,):",data["Installs"].value_counts())

#LabelEncoding
le=LabelEncoder()
data["Installs"]=le.fit_transform(data["Installs"])

#Plotting
sns.regplot(x="Installs",y="Rating",data=data).set_title("Rating vs Installs [RegPlot]")




#Code ends here



# --------------
#Code starts here
print("Value counts of price:\n",data["Price"].value_counts())

data["Price"]=data["Price"].str.replace("$","").astype(float)

sns.regplot(x="Price",y="Rating",data=data).set_title("Rating Vs Price [RegPlot]")




#Code ends here


# --------------

#Code starts here
#Print Unique
print(data["Genres"].unique())

#Split and store onlt the first Genre
data["Genres"]=data["Genres"].str.split(";").str[0]

#Groupby Genre and mean of Rating on DF(data)
gr_mean=data.groupby(["Genres"],as_index=False)["Rating"].mean()

print(gr_mean.describe())
#Sort values to find min and max
gr_mean=gr_mean.sort_values(by=["Rating"])

print("Lowest Average Rating(Genrewise):",gr_mean.iloc[0:1,])
print("Highest Average Rating(Genrewise):",gr_mean.iloc[-1:,])


#Code ends here


# --------------

#Code starts here
#print("Last Updated",data["Last Updated"])
data["Last Updated"]=pd.to_datetime(data["Last Updated"])
#print("data[Last Updated]:",data["Last Updated"])
max_date=data["Last Updated"].max()

data["Last Updated Days"]=(max_date-data["Last Updated"]).dt.days

sns.regplot(x="Last Updated Days",y="Rating",data=data).set_title("Rating vs Last Updated [RegPlot]")






#Code ends here


