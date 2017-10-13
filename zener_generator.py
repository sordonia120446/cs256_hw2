'''
Generate 25x25 Zener Cards as png files.

:authors Jason, Nick, Sam
'''

import os
import argparse
import random

from PIL import Image, ImageDraw, ImageOps

# Pos/neg in either direction
MAX_SIZE_OFFSET = 5
MAX_POS_OFFSET = 5
MAX_ROTATION = 180

DRAW_NOISE = True

def draw_shape(bg, shape, pos_offset=0, size_offset=0, rotation=0):
    '''
    Draw a Zener Card shape.

    :param bg: Background image to be drawn to
    :param shape: Shape to draw, e.g. 'O', 'P', 'Q', 'S', 'W'
    :param pos_offset: Amount to change shape position
    :param size_offset: Amount to change shape size
    :param rotation: Amount to rotate shape
    '''

    file_path = os.path.join(os.getcwd(), 'zener_shapes', shape + '.jpg')

    try:
        src = Image.open(file_path)
    except IOError:
        raise Exception('Shape not found')

    mask = ImageOps.invert(src).rotate(rotation).resize((bg.size[0] + size_offset, bg.size[1] + size_offset)).convert('1')

    bg.paste(0, box=((bg.size[0] - mask.size[0]) / 2 + pos_offset, (bg.size[1] - mask.size[1]) / 2 + pos_offset), mask=mask)

def draw_noise(im, density=0.02, iterations=50):
    '''
    Draws noise (ellipsoids) at random points on the image with a given probability.

    :param im: The image to draw the noise on
    :param density: The probability that an ellipsoid will be drawn
    :param iterations: The number of times to run the noise algorithm
    '''

    draw = ImageDraw.Draw(im)

    for n in range(0, iterations):
        if random.random() <= density:
            x1 = random.randint(0, im.size[0])
            y1 = random.randint(0, im.size[1])

            x2 = x1 + random.randint(1, 3)
            y2 = y1 + random.randint(1, 3)

            draw.ellipse((x1, y1) + (x2, y2), fill=0, outline=0)

def generate_zener_cards(args):
    '''
    Generate nny number of Zener Cards.

    :param folder_name: The name of the output folder
    :param num_examples: The number of training examples to generate
    '''

    path = os.path.join(os.getcwd(), args.folder_name)

    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))

    shapes = ['O', 'P', 'Q', 'S', 'W']
    for n in range(0, args.num_examples):
        card = Image.new('L', (25, 25), 255)

        size_offset = random.randint(-MAX_SIZE_OFFSET, MAX_SIZE_OFFSET)
        pos_offset = random.randint(-MAX_POS_OFFSET, MAX_POS_OFFSET)
        rotation = random.randint(-MAX_ROTATION, MAX_ROTATION)

        shape = random.choice(shapes)
        draw_shape(card, shape, pos_offset=pos_offset, size_offset=size_offset, rotation=rotation)

        if DRAW_NOISE and random.randint(0, 1):
            draw_noise(card)

        filename = '{}_{}.png'.format(n + 1, shape)
        card.save(os.path.join(path, filename))


# CLARGS
parser = argparse.ArgumentParser(
    description='Generate a number of 25x25 Zener cards.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

parser.add_argument(
    'folder_name',
    help='The name of the output folder.'
)
parser.add_argument(
    'num_examples',
    help='The number of images to generate.',
    type=int
)


if __name__ == '__main__':
    args = parser.parse_args()

    generate_zener_cards(args)