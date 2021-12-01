from multiprocessing import Pool
from functools import partial

'''
多进程模版
pool.map(func, iterable[, chunksize]) func只能一个可迭代参数传递进去，当需要传递多个参数时，可以将参数放到一个list或者tuple里作为一个参数传入func中
另外一种是使用偏函数(Partial function)，是通过将一个函数的部分参数预先绑定为某些值，从而得到一个新的具有较少可变参数的函数。
def func(texts, lock, data):
    pass
pt = partial(func, tests, lock)  

新函数pt可以只传入一个参数data   

'''

def fun(item):
    x, y = item
    return x * y

if __name__ == '__main__':
    pool = Pool(processes=8)
    print(pool.map(fun, list(zip(range(20), range(20)))))
    pool.close() # 关闭pool，使其不在接受新的任务。
    pool.join() # 主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。