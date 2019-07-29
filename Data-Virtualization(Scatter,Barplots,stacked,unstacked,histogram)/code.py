# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(path)
loan_status=data["Loan_Status"].value_counts()
ax=loan_status.plot(kind="bar")

plt.show()
#Code starts here


# --------------
#Code starts here
property_and_loan=data.groupby(["Property_Area","Loan_Status"])
property_and_loan=property_and_loan.size().unstack()
ax=property_and_loan.plot(kind="bar")
ax.set_xlabel("Property Area",rotation=45)
ax.set_ylabel("Loan Status")
plt.show()



# --------------
#Code starts here
education_and_loan=data.groupby(["Education","Loan_Status"])
education_and_loan=education_and_loan.size().unstack()
education_and_loan.plot(kind="bar",stacked=True,figsize=(10,15))

plt.xlabel("Education")
plt.ylabel("Loan Status")
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here

graduate=data[data['Education'] == "Graduate"]
not_graduate=data[data['Education'] == "Not Graduate"]

LoanAmount=graduate.plot.density(label="Graduate")
plt.legend()
plt.show()
LoanAmount=not_graduate.plot(label="Not Graduate")

plt.legend()
plt.show()

#Code ends here
#For automatic legend display



# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(nrows=3,ncols=1)
ax_1.scatter(x=data["ApplicantIncome"],y=data["LoanAmount"])
ax_1.set_title("Applicant Income")

ax_2.scatter(x=data["CoapplicantIncome"],y=data["LoanAmount"])
ax_2.set_title("Coapplicant Income")

data["TotalIncome"]=data["ApplicantIncome"]+data["CoapplicantIncome"]

ax_3.scatter(x=data["TotalIncome"],y=data["LoanAmount"])
ax_3.set_title("Total Income")


