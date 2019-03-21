# pumoxl

python package for use in excel via xlwings

setup instructions (Windows or Mac)
1. Download & Install Anaconda
2. run `python setup.py` to download required data
2. create environment xlenv 
3. install xlwings `conda install xlwings`
4. Install full add-in `xlwings addin install`
5. Open Excel:  
6. Goto Developer Tab -> Excel Add-Ins -> Browse, add : `C:\Users\dan\AppData\Roaming\Microsoft\Excel\XLSTART\`
  Enable Developer Tab using File -> Options -> Customize Ribbon
7. Goto `xlwings` ribbon. paste `C:\Users\dan\Anaconda3\envs\xlenv\Python.exe` into `Intepreter:`
8. Press `Alt+F11` to open vba editor. choose `xlwings` and press ok 


run the spreadsheet instructions:
double click `run_excel.bat` or `run_excel_32.bat`

Open the file : 
`prototype.xlsm`
