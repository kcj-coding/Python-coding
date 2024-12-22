import win32com.client # to interact with windows applications
import pandas as pd
import re
import os
import time

# see the below links also
# https://www.reddit.com/r/learnpython/comments/gwgtn4/what_do_you_use_win32com_for
# https://pythonexcels.com/python/2009/10/05/python-excel-mini-cookbook

###############################################################################
####### open existing excel file and convert file type ##############

# folder location
folder = r"C:\Users\david\Desktop"

# get sub-folder
folders = [x[0] for x in os.walk(folder)]

# list relevant files
files = pd.DataFrame()
for folder_name in folders:
    file = os.listdir(folder_name)
    files_x = pd.DataFrame({'folder':folder_name, 'files':file})
    files = pd.concat([files,files_x])

files = files.reset_index()
files = files.drop(columns={'index'})

# convert files
for i in range(0, len(files)):
    fname = str(files['folder'][i])+"/"+str(files['files'][i])
    newname = str(files['folder'][i])+"/"+"ABC_"+str(files['files'][i])
    
    # load the file and save as xlsx from xls
    excel = win32com.client.Dispatch("Excel.Application")
    
    wb = excel.Workbooks.Open(fname)
    
    wb.SaveAs(newname+"xlsx", FileFormat = 51) # 51 for xlsx, 56 is xls
    wb.Close()
    excel.Application.quit()

########## take converted file and send in e-mail ##################

# get files to attach in e-mail
attachments = []
for file in os.listdir(folder_name):
    if file.endswith(".xlsx"):
    # alternative if re.extract("(\.)(?=\w+$).*",file) == ".xlsx":
        filer = str(folder_name)+"/"+str(file)
        attachments.append(filer)

# draft e-mail
outlook = win32com.client.Dispatch("Outlook.Application")
mail = outlook.CreateItem(0x0)
mail.To = "email1@email.com"
mail.CC = ""
mail.Subject = "E-mail title"
mail.Body = "Email text."

for attachment in attachments:
    mail.Attachments.Add(rf"{attachment}")
    
mail.Save() # save as draft e-mail
#mail.Display() # open e-mail to send, but do not send it
#mail.Send() # send e-mail, if security permissions allow
time.sleep(0.5)



###############################################################################
########## outlook e-mail with multiple attachments ###########################

# set e-mail message variables
month = "Dec-24"

# create/read an e-mail distribution list
email_list = ["email1@email.com", "email2@email.com"]

email_df = pd.DataFrame({'email':email_list})

# generate a draft email
outlook = win32com.client.Dispatch("Outlook.Application")
for i in range(0, len(email_df)):
    flt_df = email_df.iloc[i]
    print("email "+str(i+1)+" of "+str(len(email_df)))
    
    mail = outlook.CreateItem(0x0)
    mail.To = str(flt_df['email'])
    mail.CC = ""
    mail.Subject = "E-mail title"
    #mail.Body = str("E-mail "+str(i))
    mail.HTMLBody = str("<b>E-mail</b> "+'<p style="color: red;">'+str(i)+"</p>")
    
    attachments = ["file1.pdf", "file2.csv"]
    
    for attachment in attachments:
        mail.Attachments.Add(attachment)
        
    mail.Save() # save as draft e-mail
    #mail.Display() # open e-mail to send, but do not send it
    #mail.Send() # send e-mail, if security permissions allow
    time.sleep(0.5)
    
    
###############################################################################

# functions from https://www.reddit.com/r/learnpython/comments/gwgtn4/what_do_you_use_win32com_for

# open password protected excel and save it unprotected
import win32com.client as win32
def unprotect_xlsx(filename, pw_str):
    xl = win32.Dispatch("Excel.Application")
    wb = xl.Workbooks.Open(filename, False, False, None, pw_str)
    xl.DisplayAlerts = False
    wb.SaveAs(filename, None,'','')
    xl.Quit()

# refresh all connections
import win32com.client as win32
def refresh_spreadsheet(book_name):
    xl = win32.DispatchEx("Excel.Application")
    xl.DisplayAlerts = False
    wb = xl.Workbooks.Open(book_name)
    xl.Visible = True
    wb.RefreshAll
    xl.DisplayAlerts = True
    wb.Save()
    wb.Close(True)
    
# open xlsb save as xlsx
import win32com.client as win32

##Earlier code setting variable values

excel = win32.DispatchEx('Excel.Application')
wb = excel.Workbooks.Open(os.path.join(Source_Folder, SourceFile))
#file format 51 is xlsx. This code would work to convert from any excel format (xls, xlsb, etc) to xlsx
wb.SaveAs(os.path.join(Pop_Folder, DestFile), FileFormat = 51)
excel.Application.Quit()

## continuation of script using the new file
