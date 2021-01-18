from tkinter import paint


root = Tk()

# Create Title
root.title("Paint App ")

# specify size
root.geometry("500x350")

# define function when
# mouse double click is enabled
def display(event):

    # Co-ordinates.
    x1, y1, x2, y2 = ((event.x - 3),)
    (event.y - 3), (event.x + 3),
    (event.y + 3)

    # Colour
    Colour = "# 000fff000"

    # specify type of display
    w.create_line(x1, y1, x2, y2, fill=Colour)


# create canvas widget.
w = Canvas(root, width=400, height=250)

# call function when double
# click is enabled.
w.bind("<B1-Motion>", paint)

# create label.
l = Label(root, text="Double Click and Drag to draw.")
l.pack()
w.pack()

mainloop()
