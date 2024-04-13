import turtle
import random
n=int(input("number of iterations of the dragon curve (give less than 10 ):-"))
#try to give the iterations as even numbers to see symmetry
# powers of 2 are more prefereable
step=float(input("step length for forward propagation :- "))
#adjust this step value to increase the size of the curve
#note as you increase the number of intersections , decrease the forward step value
"""
L system used :- 
variables : F G
constants : + -
start  : F
rules  : (F → F-G), (G → F+G)
angle  : 90°
Here, F and G both mean "draw forward", + means "turn right by angle", and - means "turn left by angle". 
"""
def generate_string(n):
    if n == 0:
        return "F"
    else:
        prev_string = generate_string(n-1)
        new_string = ""
        for char in prev_string:
            if char == "F":
                new_string += "F-G"
            elif char == "G":
                new_string += "F+G"
            else:
                new_string += char
        return new_string
draw_string=generate_string(n)
print(len(draw_string))
turtle.penup()
turtle.goto(0, 0)#starting pos of turtle to draw the curve
turtle.pendown()
turtle.speed(10000000000000)#speed of drawing can be still increased
for i in draw_string:
    if i=='F' or i=='G':
        color = (random.random(), random.random(), random.random())  # Random RGB color
        turtle.color(color)
        turtle.forward(step)
    elif i=='+':
        turtle.left(90)# - sign to denote anticlockwise direction
    elif i=='-':
        turtle.right(90)
turtle.done()
    
