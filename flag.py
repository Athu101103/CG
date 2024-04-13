import turtle

def draw_rectangle(color, x, y, width, height):
    turtle.penup()
    turtle.goto(x, y)
    turtle.fillcolor(color)
    turtle.begin_fill()
    for _ in range(2):
        turtle.forward(width)
        turtle.right(90)
        turtle.forward(height)
        turtle.right(90)
    turtle.end_fill()

# Set up the screen
screen = turtle.Screen()
screen.setup(width=600, height=400)
screen.bgcolor("#FFFFFF")

# Draw the saffron rectangle
draw_rectangle("#FF9933", -300, 150, 600, 100)

# Draw the white rectangle
draw_rectangle("#FFFFFF", -300, 50, 600, 100)

# Draw the green rectangle
draw_rectangle("#138808", -300, -50, 600, 100)

def draw_chakra(x, y, radius, num_spokes):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.color("#000080")
    turtle.begin_fill()
    turtle.circle(radius)
    turtle.end_fill()
    angle = 360 / num_spokes
    for _ in range(num_spokes):
        turtle.penup()
        turtle.goto(x, y)
        turtle.setheading(_ * angle)
        turtle.forward(radius)
        turtle.pendown()
        turtle.forward(radius)
        turtle.penup()

draw_chakra(0 , 0, 50, 24)

# Draw the navy blue circle
turtle.penup()
turtle.goto(0, -50)
turtle.pendown()
turtle.color("#000080")
turtle.begin_fill()
turtle.circle(50)
turtle.end_fill()

# Hide the turtle and display the result
turtle.hideturtle()
turtle.done()
