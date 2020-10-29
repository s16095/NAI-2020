#Zadanie ASCII ART

#URL Zadania:
#https://www.codingame.com/ide/puzzle/ascii-art

#Autor:
#Adrian Wojewoda s16095

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

l = int(input())
h = int(input())
t = input()
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
word = t.upper()
beyondAlph = []
for w in word:
    for a in alphabet:
        if w in alphabet:
            pass
        else:
            beyondAlph.append(w)
            break
beyondAlph = list(dict.fromkeys(beyondAlph))
for i in range(h):
    row = input()
    phrase = ""
    for j in word:
        if j in beyondAlph: #FILTER AUTOMATICALLY
            pos = alphabet.find(alphabet[-1]) + 1
        else:
            pos = alphabet.find(j)
        phrase += row[l*pos: l*pos + l]
    print(phrase)