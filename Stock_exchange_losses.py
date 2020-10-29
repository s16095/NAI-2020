#Zadanie Stock exchange losses

#URL Zadania:
#https://www.codingame.com/ide/puzzle/stock-exchange-losses

#Autor:
#Adrian Wojewoda s16095

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

n = int(input())
values = []
loss = []
for i in input().split():
    v = int(i)
    values.append(v)

reff = values[0]
for j in range(1,n):
    loss.append(-reff + values[j])
    if loss[j - 1] > 0:
        reff = values[j]
if min(loss) >= 0:
    print(0)
elif min(loss) < 0:
    print(min(loss))