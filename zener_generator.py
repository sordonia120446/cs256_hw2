import os, sys
import random

from PIL import Image, ImageDraw

offset = 4.5 # Should not be zero

def draw_circle(im, rotation=0, stroke=1):
    shape = Image.new('1', (im.size[0], im.size[1]), 0)
    x_off, y_off = shape.size[0] / offset, shape.size[1] / offset

    draw = ImageDraw.Draw(shape)
    draw.ellipse((x_off, y_off) + (shape.size[0] - x_off, shape.size[1] - y_off), fill=0, outline=1)
    del draw

    im.paste(0, mask=shape.rotate(0))

def draw_cross(im, rotation=0, stroke=1):
    shape = Image.new('1', (im.size[0], im.size[1]), 0)
    x_off, y_off = shape.size[0] / offset, shape.size[1] / offset

    draw = ImageDraw.Draw(shape)
    draw.line((shape.size[0] / 2., y_off) + (shape.size[0] / 2., shape.size[1] - y_off), fill=1, width=stroke)
    draw.line((x_off, shape.size[0] / 2.) + (shape.size[0] - x_off, shape.size[1] / 2.), fill=1, width=stroke)
    del draw

    im.paste(0, mask=shape.rotate(rotation))

def draw_square(im, rotation=0, stroke=1):
    shape = Image.new('1', (im.size[0], im.size[1]), 0)
    x_off, y_off = shape.size[0] / offset, shape.size[1] / offset

    draw = ImageDraw.Draw(shape)
    draw.line((x_off, y_off) + (shape.size[0] - x_off, y_off), fill=1, width=stroke)
    draw.line((shape.size[0] - x_off, y_off) + (shape.size[0] - x_off, shape.size[1] - y_off), fill=1, width=stroke)
    draw.line((shape.size[0] - x_off, shape.size[1] - y_off) + (x_off, shape.size[1] - y_off), fill=1, width=stroke)
    draw.line((x_off, shape.size[1] - y_off) + (x_off, y_off), fill=1, width=stroke)
    del draw

    im.paste(0, mask=shape.rotate(rotation))

def generate_zener_cards(folder_name, num_examples):
    path = os.path.join(os.getcwd(), folder_name)

    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))

    for i in range(0, num_examples):
        card = Image.new('1', (25, 25), 1)

        choice = random.randint(1, 3)
        if choice == 1:
            draw_circle(card)
            card_type = 'O'
        elif choice == 2:
            draw_cross(card)
            card_type = 'P'
        else:
            draw_square(card)
            card_type = 'Q'

        filename = '{}_{}.png'.format(i + 1, card_type)
        card.save(os.path.join(path, filename))

if len(sys.argv) != 3:
    raise Exception('Incorrect number of parameters')
else:
    generate_zener_cards(sys.argv[1], int(sys.argv[2]))