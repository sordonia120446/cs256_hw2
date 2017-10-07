import os, sys
import random

from PIL import Image, ImageOps

MAX_SIZE_OFFSET = 10
MAX_POS_OFFSET = 5
MAX_ROTATION = 360
MAX_NOISE_PROBABILITY = 0.5

def draw_shape(bg, shape, pos_offset=0, size_offset=0, rotation=0):
    file_path = os.path.join(os.getcwd(), 'zener_shapes', shape + '.jpg')

    try:
        src = Image.open(file_path)
    except IOError:
        print 'Shape not found'

    mask = ImageOps.invert(src).convert('1').rotate(rotation).resize((bg.size[0] - size_offset, bg.size[1] - size_offset))

    bg.paste(0, box=((bg.size[0] - mask.size[0]) / 2 + pos_offset, (bg.size[1] - mask.size[1]) / 2 + pos_offset), mask=mask)

def generate_zener_cards(folder_name, num_examples):
    path = os.path.join(os.getcwd(), folder_name)

    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))

    shapes = ['O', 'P', 'Q', 'S', 'W']
    for n in range(0, num_examples):
        card = Image.new('1', (25, 25), 1)

        size_offset = random.randint(0, MAX_SIZE_OFFSET)
        pos_offset = random.randint(-MAX_POS_OFFSET, MAX_POS_OFFSET)
        rotation = random.randint(0, MAX_ROTATION)

        shape = random.choice(shapes)
        draw_shape(card, shape, pos_offset=pos_offset, size_offset=size_offset, rotation=rotation)

        filename = '{}_{}.png'.format(n + 1, shape)
        card.save(os.path.join(path, filename))

if len(sys.argv) != 3:
    raise Exception('Incorrect number of parameters')
else:
    generate_zener_cards(sys.argv[1], int(sys.argv[2]))