# visualizing the company's record with respect to loan approvals. --------------
#Importing header files
import  visualizing the company's record with respect to loan approvals.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(path)
loan_status=data["Loan_Status"].value_counts()
ax=loan_status.plot(kind="bar")

plt.show()
#Code starts here


# --------------
#Code starts here
#Plotting an unstacked bar plot
property_and_loan=data.groupby(["Property_Area","Loan_Status"])
property_and_loan=property_and_loan.size().unstack()
ax=property_and_loan.plot(kind="bar")
#Changing the x-axis label
ax.set_xlabel("Property Area",rotation=45)
#Changing the y-axis label
ax.set_ylabel("Loan Status")
plt.show()



# does higher education result in a better guarantee in issuing loans?--------------
#Code starts here
#Plotting a stacked bar plot
education_and_loan=data.groupby(["Education","Loan_Status"])
education_and_loan=education_and_loan.size().unstack()
education_and_loan.plot(kind="bar",stacked=True,figsize=(10,15))

plt.xlabel("Education")
plt.ylabel("Loan Status")
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here

#Subsetting the dataframe based on 'Education' column
graduate=data[data['Education']=='Graduate']


#Subsetting the dataframe based on 'Education' column
not_graduate=data[data['Education']=='Not Graduate']


#Plotting density plot for 'Graduate'
graduate['LoanAmount'].plot(kind='density', label='Graduate')


#Plotting density plot for 'Graduate'
not_graduate['LoanAmount'].plot(kind='density',label='Not Graduate')

plt.legend()
plt.show()
#Code ends here




# Income Vs Loan--------------
#Code starts here

#Setting up the subplots
fig, (ax_1, ax_2,ax_3) = plt.subplots(1,3, figsize=(20,8))

#Plotting scatter plot
ax_1.scatter(data['ApplicantIncome'],data["LoanAmount"])

#Setting the subplot axis title
ax_1.set(title='Applicant Income')


#Plotting scatter plot
ax_2.scatter(data['CoapplicantIncome'],data["LoanAmount"])

#Setting the subplot axis title
ax_2.set(title='Coapplicant Income')


#Creating a new column 'TotalIncome'
data['TotalIncome']= data['ApplicantIncome']+ data['CoapplicantIncome']

#Plotting scatter plot
ax_3.scatter(data['TotalIncome'],data["LoanAmount"])

#Setting the subplot axis title
ax_3.set(title='Total Income')


#Code ends here
