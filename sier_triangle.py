import turtle
import random
n=int(input("number of iterations of the sier-triangle (give less than 10 ):-"))
#try to give the iterations as even numbers to see symmetry
# powers of 2 are more prefereable
step=float(input("step length for forward propagation :- "))
#adjust this step value to increase the size of the curve
#note as you increase the number of intersections , decrease the forward step value

"""
L system used :- 
variables : A B
constants : + -
start  : A
rules  : (A → B-A-B), (B → A+B+A)
angle  : 60°
Here, A and B both mean "draw forward", + means "turn left by angle", and - means "turn right by angle" 
"""
def generate_string(n):
    if n == 0:
        return "A"
    else:
        prev_string = generate_string(n-1)
        new_string = ""
        for char in prev_string:
            if char == "A":
                new_string += "B-A-B"
            elif char == "B":
                new_string += "A+B+A"
            else:
                new_string += char
        return new_string
draw_string=generate_string(n)
print(len(draw_string))
turtle.penup()
turtle.goto(0, 0)#starting pos of turtle to draw the curve
turtle.pendown()
turtle.speed(1000000)#speed of drawing can be still increased
for i in draw_string:
    if i=='A' or i=='B':
        color = (random.random(), random.random(), random.random())  # Random RGB color
        turtle.color(color)
        turtle.forward(step)
    elif i=='+':
        turtle.left(60)
    elif i=='-':
        turtle.right(60)
turtle.done()
    
