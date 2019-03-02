import xlwings as xw
import matplotlib.pyplot as plt 
import datetime 
import numpy as np

def hello_xlwings():
    wb = xw.Book.caller()
    wb.sheets[0].range("A1").value = "Hello xlwings!"


@xw.func
def hello(name):
    return "hello {0}".format(name)

@xw.func
def myplot(n):
    sht = xw.Book.caller().sheets.active
    fig = plt.figure()
    plt.plot(range(int(n)))
    sht.pictures.add(fig,name = 'Plot', update = True)
    return 'Plotted with n={}'.format(n)

@xw.func
def plot_ts(ts):
    sht = xw.Book.caller().sheets.active
    fig = plt.figure()
    plt.plot(ts)
    sht.pictures.add(fig,name = 'Plot', update = True)
    return 'Plotted with {} rows'.format(len(ts))

@xw.func
@xw.ret(index = False,header= True,expand = 'table')
def simulate_CER_Price(starting_price =20, t = 240,num_paths = 25, start_date =datetime.datetime.now ):
    CER_Prices = np.zeros((t,num_paths))
    for i in range(num_paths):
        forecast_returns = np.random.normal(.005, .112, t)
        forecast_prices  = starting_price * np.cumprod(1+forecast_returns)
        CER_Prices[:,i] =forecast_prices.transpose()
    return CER_Prices

@xw.func
@xw.ret(index = False,header= False,expand = 'table')
def test_array_func():
    return [1,2,3,4,5]

#debug server
#if __name__ == "__main__":
#   xw.serve()