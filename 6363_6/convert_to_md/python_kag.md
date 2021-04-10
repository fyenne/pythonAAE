<center><span style="font-size:42px;color:#ff3399;">Python data science <br> for primary school students </center></span>



source from programiz.com;  calmcode.io; kaggle.com;

`by Siming Yan` @ www.fyenneyenn.studio

同步更新 @ https://fyenneyenn.gitee.io/pythonaae

---

# install

本文选择使用`miniconda + pycharm` 或者 `miniconda + visual studio code`

如果已经安装了python, 有python基础, 或者已经安装了 anaconda等ide可以略过第一部分

## miniconda

`本部分参考自 https://www.cnblogs.com/ruhai/p/10847220.html`

a. Download:

`https://docs.conda.io/en/latest/miniconda.html`  官网

`https://repo.anaconda.com/miniconda/` 或者通过清华大学镜像

选择适合自己的版本进行安装: 如mac系统, 选择(64 pkg 文件)完成快速安装 

| Python version | Name                                                         | Size     | SHA256 hash                                                  |
| -------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| Python 3.9     | [Miniconda3 MacOSX 64-bit bash](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh) | 42.2 MiB | `b3bf77cbb81ee235ec6858146a2a84d20f8ecdeb614678030c39baacb5acbed1` |
|                | [Miniconda3 MacOSX 64-bit pkg](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-MacOSX-x86_64.pkg) | 49.7 MiB | `298ff80803817921a98e21d81d60f93b44afce67aec8ae492d289b13741bcffe` |
| Python 3.8     | [Miniconda3 MacOSX 64-bit bash](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh) | 54.5 MiB | `a9ea0afba55b5d872e01323d495b649eac8ff4ce2ea098fb4c357b6139fe6478` |
|                | [Miniconda3 MacOSX 64-bit pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg) | 62.0 MiB | `b06f3bf3cffa9b53695c9c3b8da05bf583bc7047d45b0d74492f154d85e317fa` |

b. 验证安装 

打开terminal(终端):

输入`conda -V` 验证安装, 注意大写V.

c. 添加miniforge到环境变量:

打开terminal, 输入:

```bash
sudo nano /etc/paths
# 输入你的开机密码; 按回车
# 在terminal的文本编辑器里继续输入
# /Users/你的电脑用户名字/miniforge3/bin 如:狗蛋儿
#按esc 
#输入 
#:wq 
#以退出
#记得保存
```

---

因为很多包开发于国外, 下载速度慢, 所以在terminal中建议粘贴以下命令, 使得默认下载链接为国内的镜像地址.

ps: 中科大的比清华的好用, 可以把清华的删了. (清华的同学请忽视本ps)

```bash
# 清华的镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

# 中国科技大学源
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/

# 阿里云的服务器
# http://mirrors.aliyun.com/pypi/simple/ 
# 豆瓣网站是用python做的呀 豆瓣的python的源
# conda config --add channels http://pypi.douban.com/simple/ 
```

```bash
# 两个基本的设置, 可以自行选择是否添加.
# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
# 当安装包时，总会被询问是否`Proceed ([y]/n)?` ，默认为False，设置为`always_yes: True`将不会再做提醒
conda config --set always_yes True
```

---

创造虚拟环境以_**科学**_地使用python: 因为你直接用电脑自带的环境我怕你弄坏了电脑都废了. 虚拟环境坏了还可以删库跑路就当没学过.

```bash
# 打开 terminal 输入
# 将 你起个名字 替换成任意英文名字
conda create -n 你起个名字 python=3.8
# 验证虚拟环境是否成功搭建
conda info --envs # 输出虚拟环境目录
```

```bash
# base                  	 *  /Users/你的用户名字/miniforge3
# 你起个名字                   /Users/你的用户名字/miniforge3/envs/你起个名字
```

启动环境

```bash
conda activate 你起个名字 # 启动环境
python -V # 验证python
# 输出为
# Python 3.8.6 
```

安装成功! 

# vscode or pycharm or even R studio

+ 下载vscode

`https://code.visualstudio.com/Download` 

+ 或者学生可以使用免费的一年期限的pycharm professional

`https://www.jetbrains.com/pycharm/download/#section=mac`

+ 想要在Rstudio中使用python conda environment

需要通过conda 重新安装R; 

`conda install R` 设置比较麻烦, 不做赘述.



*建议使用`vscode`因为主题比较好看.*

---

在vscode中安装python 插件, jupyter 插件

![Screen Shot 2021-04-10 at 10.10.45 PM](/Users/fyenne/Downloads/booooks/semester5/pythonAAE/pics/Screen Shot 2021-04-10 at 10.10.45 PM.jpg)

在vscode中按下`cmd+shift+p`输入jupyter create 后按下回车新建一个ipynb的文件.

如此, 我们有了第一个python的脚本. 

![Screen Shot 2021-04-10 at 10.40.35 PM](/Users/fyenne/Downloads/booooks/semester5/pythonAAE/pics/Screen Shot 2021-04-10 at 10.40.35 PM.png)

输入` print('哈咯握的')` 按下shift+回车运行该chunk

# input and output.


```python
print(1+1) #  
print(3/2); 
print(3//2); # 取整
print(6%2) # 取余
print(6**3) # power
(1.5).as_integer_ratio()
# print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```

    2
    1.5
    1
    0
    4

```python
x = 3; y = 'S'
print('The value of x is {} and y is {}'.format(x,y))
x = 12.345678900
print('%.2d' %x) # or i, integer.
print('%.2f' %x) # float with 2
print('%.4f' %x) # float with 4
print('%.4e' %x) # scientific format
print('%4u'  %x) # with indent
print('%.4u' %x) # with number indent
print( '%c'  %y) # one character

z = 'zhen\de\t\hen\nan'
print( '%s'  %z) # a string # \t == space, \n == new line
print('#-------------------------------------------')
print(r'zhen\de\t\hen\nan', '\n', r'%s' %z)
```

    The value of x is 3 and y is S
    12
    12.35
    12.3457
    1.2346e+01
      12
    0012
    S
    zhen\de	\hen
    an
    #-------------------------------------------
    zhen\de\t\hen\nan
    zhen\de	\hen
    an

```python
hw12 = '%s %s %d' % ('hello this', 'world', 12)  # sprintf style string formatting
print(hw12) 
```

    hello this world 12

`字符串前加 u`
例：u"我是含有中文字符组成的字符串。"
作用：
后面字符串以 Unicode 格式 进行编码，一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码。
`字符串前加 r`
例：r"\n\n\n\n”　　

`# 表示一个普通raw data 生字符串 \n\n\n\n，而不表示换行了。`
作用：
去掉反斜杠的转移机制。
(特殊字符：即那些，反斜杠加上对应字母，表示对应的特殊含义的，比如最常见的”\n”表示换行，”\t”表示Tab等。 )


```python
import math
print(math.pi)
```

    3.141592653589793

# define a function


```python
def summ(a,b):
    return a+b
```


```python
summ(3,5)
```


    8




```python
def onetopping(ketchup, onion, mayo):
    return int(ketchup) + int(onion) + int(mayo) == 1
    # values can take in this function: int
```


```python
print(onetopping(1,1,0), onetopping(0,0,1));
onetopping(1, -1, 1)
```

    False True

    True


```python
triple = lambda x: x * 3;
# this would equal to def a function
def triple2(x):
    return x * 3
```


```python
triple(3), triple2(4)
```


    (9, 12)



# List in python


```python
type([3,6,8,10,1,2,1])
```

list


```python
a = [1,2,3,4,"5"];a
```

[1, 2, 3, 4, '5']


```python
b = list(map(int, a));print(b)
```

[1, 2, 3, 4, 5]

```python
b.sort(reverse = True);b
```

[5, 4, 3, 2, 1]


```python
sorted(b, reverse = False)
```

[1, 2, 3, 4, 5]


```python
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
```


```python
planets[0:], planets[1:-1], planets[:-1]
```


    (['Mercury',
      'Venus',
      'Earth',
      'Mars',
      'Jupiter',
      'Saturn',
      'Uranus',
      'Neptune'],
     ['Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'],
     ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'])


```python
planets.append('Pluto');planets
```


    ['Mercury',
     'Venus',
     'Earth',
     'Mars',
     'Jupiter',
     'Saturn',
     'Uranus',
     'Neptune',
     'Pluto']


```python
planets.index('Earth'), planets[2]
```


    (2, 'Earth')


```python
primes = [2, 3, 5, 7]
sum(primes), max(primes)
```


    (17, 7)


```python
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]
```


```python
hands[2], hands[2][2], hands[0][0]
```


```python
%time
sum(range(100))
```

    CPU times: user 2 µs, sys: 0 ns, total: 2 µs
    Wall time: 5.01 µs

    4950


```python
# noinspection PyBroadException
def select_second(L):
    """Return the second element of the given list. If the list has no second
    element, return None.
    """
    try:
        if type(L[1]) == int or float:
            return L[1]
    except:
        return None
    pass
#--------------------------------------------
# or
def select_second2(L):
    try:
        type(L[1]) == int or float
        return L[1]
    except:
        return None
    pass
```


```python
L = [2,3,4]
select_second2(L),select_second(L)
```




    (3, 3)




```python
def quicksort(arr):
    """make a quick sort function for data lists."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

```

    [1, 1, 2, 3, 6, 8, 10]

## args, kwargs and functions


```python
def multiply(*numbers): # star marks for multiple inputs
    result = 1
    for n in numbers:
        result = result * n
    return result

multiply(1,3,5)
```


```python
def function(a, b, *args):
    # *args: a tuple; unnamed arguments in the following function
    print(a + b + sum(args))
    print(sum(args))

function(1,2,3,4)
```

    10
    7

```python
def function(a, b, *args, keyword = True, **kwargs):
    # **kwargs a dictionary; named arguments in the following function
    print(a + b + sum(args))
    print(sum(args))
    print(keyword)
    print(kwargs)
    df = kwargs
    print(df['wo'])
function(1,2,3,4, wo = 'sb')
```

    10
    7
    True
    {'wo': 'sb'}
    sb


