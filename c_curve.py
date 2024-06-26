import turtle
import random
n=int(input("number of iterations of the C curve (give less than 10 ):-"))
#try to give the iterations as even numbers to see symmetry
# powers of 2 are more prefereable
step=float(input("step length for forward propagation :- "))
#adjust this step value to increase the size of the curve
#note as you increase the number of intersections , decrease the forward step value
"""
L system used :- 
Variables:	F
Constants:	+ -
Start:	F
Rules:	F → +F--F+
where "F" means "draw forward", "+" means "turn clockwise 45°", and "-" means "turn anticlockwise 45°". 
    
"""
def generate_string(n):
    if n == 0:
        return "F"
    else:
        prev_string = generate_string(n-1)
        new_string = ""
        for char in prev_string:
            if char == "F":
                new_string += "+F--F+"
            else:
                new_string += char
        return new_string
draw_string=generate_string(n)
turtle.penup()
turtle.goto(0, 200)#starting pos of turtle to draw the curve
turtle.pendown()
turtle.speed(10000)#speed of drawing can be still increased
for i in draw_string:
    if i=='F':
        color = (random.random(), random.random(), random.random())  # Random RGB color
        turtle.color(color)
        turtle.forward(step)
    elif i=='+':
        turtle.right(45)
    elif i=='-':
        turtle.left(45)
turtle.done()
    
