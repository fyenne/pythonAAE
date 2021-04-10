<center><span style="font-size:42px;color:#ff3399;">Python data science <br> for primary school students </center></span>



source from programiz.com; kaggle.com; calmcode.io

`by Siming Yan` @ www.fyenneyenn.studio



---

# input and output.


```python
print(1+1)
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
x = 3; y = 'S'; z = 'zhen\de\t\hen\nan'
print('The value of x is {} and y is {}'.format(x,y))
x = 12.345678900
print('%.2d' %x) # or i, integer.
print('%.2f' %x) # float with 2
print('%.4f' %x) # float with 4
print('%.4e' %x) # scientific format
print('%4u'  %x) # with indent
print('%.4u' %x) # with number indent
print( '%c'  %y) # one character
print( '%s'  %z) # a string # \t == space, \n == new line
print('#-------------------------------------------')
print(r'zhen\de\t\hen\nan', '---', r'%s' %z)
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
    zhen\de\t\hen\nan --- zhen\de	\hen
    an

```python
hw12 = '%s %s %d' % ('hello this', 'world', 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

    hello this world 12

`字符串前加 u`
例：u"我是含有中文字符组成的字符串。"
作用：
后面字符串以 Unicode 格式 进行编码，一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码。
`字符串前加 r`
例：r"\n\n\n\n”　　# 表示一个普通生字符串 \n\n\n\n，而不表示换行了。
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



```python

```
