import turtle
import random
n=int(input("number of iterations of the terrain (give less than 10 ):-"))
#try to give the iterations as even numbers to see symmetry
# powers of 2 are more prefereable
step=float(input("step length for forward propagation :- "))
#adjust this step value to increase the size of the curve
#note as you increase the number of intersections , decrease the forward step value

"""
L system used :- 
variables : F
constants : + *
start  : F
rules  : (F -> F+F*F+F)
angle  : 60°,120°
Here, F means "draw forward", + means "turn left by angle" , and * means turn right by 120 deg
"""
def generate_string(n):
    if n == 0:
        return "F"
    else:
        prev_string = generate_string(n-1)
        new_string = ""
        for char in prev_string:
            if char == "F":
                new_string += "F+F*F+F"
            else:
                new_string += char
        return new_string
draw_string=generate_string(n)
print(len(draw_string))
turtle.penup()
turtle.goto(-150, -150)#starting pos of turtle to draw the curve
turtle.pendown()
turtle.speed(1000000)#speed of drawing can be still increased
for i in draw_string:
    if i=='F':
        color = (random.random(), random.random(), random.random())  # Random RGB color
        turtle.color(color)
        turtle.forward(step)
    elif i=='+':
        turtle.left(60)
    elif i=='*':
        turtle.right(120)
turtle.done()
    
