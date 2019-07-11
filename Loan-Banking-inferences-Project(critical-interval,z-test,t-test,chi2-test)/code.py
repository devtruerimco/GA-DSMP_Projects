# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data=pd.read_csv(path)
#Code starts here
data_sample=data.sample(n=sample_size,random_state=0)

sample_mean=data_sample["installment"].mean()

sample_std=data_sample["installment"].std()

margin_of_error=z_critical*(sample_std/(sample_size)**0.5)

confidence_interval=sample_mean-margin_of_error,sample_mean+margin_of_error

print("CI:",confidence_interval)
true_mean=data["installment"].mean()

print("true mean:",true_mean)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig,axes=plt.subplots(nrows=3,ncols=1)

for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        m.append(data["installment"].sample(n=sample_size[i]).mean())
    mean_series=pd.Series(m)
    axes[i].hist(mean_series)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
#print(data["int.rate"])
data["int.rate"]=data["int.rate"].str.replace("%","")
data["int.rate"]=data["int.rate"].astype(float)
data["int.rate"]=data["int.rate"]/100

z_statistic,p_value=ztest(x1=data[data["purpose"]=="small_business"]["int.rate"],value=data["int.rate"].mean(),alternative="larger")

print("z_statistic",z_statistic)
print("p_value",p_value)

if p_value<0.05:
    print("Reject H0,Interest rate being given to people with purpose as 'small_business' is higher than the average interest rate")
else:
    print("Accept H0,There is no difference in interest rate being given to people with purpose as 'small_business'")



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value=ztest(x1=data[data["paid.back.loan"]=="No"]["installment"],x2=data[data["paid.back.loan"]=="Yes"]["installment"])
print("z_statistic",z_statistic)
print("p_value",p_value)

if p_value<0.05:
    print("Reject H0,There is no difference in installments being paid by loan defaulters and loan non defaulters")
else:
    print("Accept H0,: There is difference in installments being paid by loan defaulters and loan non defaulters")


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data["paid.back.loan"]=="Yes"]["purpose"].value_counts()

no=data[data["paid.back.loan"]=="No"]["purpose"].value_counts()

observed=pd.concat([yes.transpose(),no.transpose()],axis=1,keys=["Yes","No"])

chi2,p,dof,ex=chi2_contingency(observed)

print("chi2:",chi2)
print("dof:",dof)
print("ex:",ex)

if chi2<critical_value:
    print("Reject H0,Distribution of purpose across all customers is same.")
else:
    print("Accept H0,Distribution of purpose for loan defaulters and non defaulters is different.")



