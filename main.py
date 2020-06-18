"""

References:
- Crop / Split / Collate PDFs https://gist.github.com/roliveira/a18f6a16754edc9caa3424d9fa1e5d6d

"""

import json
import os
import tempfile

from PyPDF2 import PdfFileWriter, PdfFileReader

import turtle


WIDTH = 5000


# Tree Methods

def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def tree_to_tokens(parse):
    if not isinstance(parse, (list, tuple)):
        return [parse]
    return tree_to_tokens(parse[0]) + tree_to_tokens(parse[1])


def tree_to_string(parse):
    if not isinstance(parse, (list, tuple)):
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        result = '( '
        for i, x in enumerate(parse):
            if i > 0:
                result += ' '
            result += tree_to_string(x)
        result += ' )'
        return result


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    _tokens = [x for x in parse.split(' ') if x not in (')', '(')]
    print('parse:', parse)
    print('tokens:', _tokens)
    counter = 0
    counter_stack = []
    buff = list(reversed(_tokens))
    stack = []

    for i, word in enumerate(parse.split(' ')):
        if word[0] == "(":
            counter_stack.append(counter)
        elif word == ")":
            pivot = counter_stack.pop()
            size = counter - pivot
            counter = pivot + 1

            substack = []
            for j in range(size):
                substack = [stack.pop()] + substack

                if j == 0:
                    pass
                elif j < size - 1:
                    transitions.append(2)
                else:
                    transitions.append(1)
            stack.append(substack)

        else:
            # Downcase all words to match GloVe.
            if lowercase:
                tokens.append(word.lower())
            else:
                tokens.append(word)
            transitions.append(0)

            stack.append(buff.pop())
            counter += 1

    print('demo:')
    print(parse)
    print(transitions)
    print(stack)

    return tokens, transitions


# Draw Methods

class Node(object):
    def __init__(self, val=None, width=None, height=None, pos=None, mid=None,
                 lx=None, rx=None, lnodeid=None, rnodeid=None):
        self.val = val
        self.width = width
        self.height = height
        self.pos = pos
        self.mid = mid
        self.lx = lx
        self.rx = rx
        self.lnodeid = lnodeid
        self.rnodeid = rnodeid


class CustomTurtle(object):
    def __init__(self, turtle):
        self.turtle = turtle
        self.history = []

    def goto(self, x, y):
        self.turtle.goto(x, y)
        self.history.append((x, y))

    def position(self):
        pos = self.turtle.position()
        self.history.append(pos)
        return pos

    def bounding_box(self):
        xs = [pos[0] for pos in self.history]
        ys = [pos[1] for pos in self.history]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        height = 500
        width = WIDTH
        pad = 10

        x0 = width/2 + xmin - pad
        x1 = width/2 + xmax + pad
        y0 = height/2 + ymin - pad
        y1 = height/2 + ymax + pad

        return dict(x0=x0, x1=x1, y0=y0, y1=y1)


class TreeFig(object):
    def __init__(self, style='box', color='#000', size=None):
        self.style = style
        self.color = color
        self.size = size
        self.turtle = None
        self.cturtle = None

    def setup_turtle(self, widthWindow, heightWindow, scale, x0):

        self.cturtle = CustomTurtle(turtle)

        turtle.setup(widthWindow, heightWindow)
        turtle.reset()
        turtle.hideturtle()
        turtle.penup()
        self.cturtle.goto(x0, 0)


    def srdraw(self, s, ts, ws, x0=0, y0=30, yMax=200, adjust_top=True, mask=None, style=None):
        n = len(s)
        offsets = [x0] + [x0 + sum(ws[:i]) for i in range(1,n)]
        buff = [Node(val=w, height=1, width=ws[n-i-1], mid=offsets[n-i-1] + ws[n-i-1]/2, pos=n-i-1, lnodeid=n-i-1, rnodeid=n-i-1) for i, w in enumerate(s[::-1])]
        sofar = 0
        stack = []

        # For n-ary.
        substack = []
        #

        if mask is None:
            mask = [0] * len(ts)
        mask = [x == 1 for x in mask]

        # Draw Vars
        interval = (yMax-y0) / (n-1)

        # SR
        for t, m in zip(ts, mask):

            x = None
            if t == 2:
                substack.append(True)
            elif t == 0: # Shift
                #
                x = buff.pop()
                sofar += 1
            elif t == 1: # Reduce
                rx = stack.pop()
                lx = stack.pop()

                # n-ary
                middle = []
                while len(substack) > 0:
                    middle.append(lx)
                    substack.pop()
                    lx = stack.pop()
                #

                xpos = min(lx.pos, rx.pos)

                lmid = lx.mid
                rmid = rx.mid
                xmid = offsets[xpos] + lx.width / 2 + (lx.width / 2 + rx.width / 2) / 2

                x = Node(val=(lx.val, rx.val),
                         height=max(lx.height, rx.height) + 1,
                         width=lx.width + rx.width,
                         pos=xpos,
                         mid=xmid,
                         lnodeid=lx.lnodeid,
                         rnodeid=rx.rnodeid,
                         )

                if adjust_top:
                    xmid = lx.mid + (rx.mid - lx.mid) / 2

                    x = Node(val=(lx.val, rx.val),
                             height=max(lx.height, rx.height) + 1,
                             width=lx.width + rx.width,
                             pos=xpos,
                             mid=xmid,
                             lx=lx,
                             rx=rx,
                             lnodeid=lx.lnodeid,
                             rnodeid=rx.rnodeid,
                             )

                ly = y0 + (lx.height-1) * interval
                ry = y0 + (rx.height-1) * interval
                xy = y0 + (x.height-1) * interval

                def boxtree():
                    turtle.penup()
                    self.cturtle.goto(lmid, ly)
                    turtle.pendown()
                    self.cturtle.goto(lmid, xy)
                    self.cturtle.goto(xmid, xy)
                    self.cturtle.goto(rmid, xy)
                    self.cturtle.goto(rmid, ry)
                    turtle.penup()

                turtle.pensize(self.size)
                turtle.pencolor(self.color)

                pensize = turtle.pensize()

                if style is not None:
                    ledgeid = style[lx.lnodeid].get('edgeid', None)
                    redgeid = style[rx.rnodeid].get('edgeid', None)
                    print(lx.lnodeid, rx.rnodeid, ledgeid, redgeid)
                    if ledgeid is not None and ledgeid == redgeid:
                        turtle.pensize(3)

                if m:
                    default = turtle.pencolor()
                    turtle.pencolor('#0376BA')

                boxtree()

                for node in middle:
                    # if node.height == 1 and x.height == 2:
                    #     continue
                    nodemid = node.mid
                    nodey = y0 + (node.height-1) * interval
                    print('node-middle', nodemid, nodey)

                    turtle.penup()
                    self.cturtle.goto(nodemid, nodey)
                    turtle.pendown()
                    self.cturtle.goto(nodemid, xy)
                    turtle.penup()

                # Reset.
                turtle.pensize(pensize)

                if m:
                    turtle.pencolor(default)

            if x is not None:
                stack.append(x)


    def draw_tree(self, parse, tokens, **settings):

        style = settings.get('style', None)

        # Size
        scale = 1
        y0 = 65 * scale
        yMax = 200 * scale
        widthWindow = WIDTH * scale
        x0 = -widthWindow/2 + 10

        def write_sentence(s):

            # Font Name
            fontname = 'Helvetica Neue'

            # Font Weight
            fontweight = ['normal'] * len(s)
            if style is not None:
                fontweight = [x.get('fontweight', 'normal') for x in style]

            prv = self.cturtle.position()[0]
            for i, w in enumerate(s):
                font = (fontname, 20 * scale, fontweight[i])
                if i > 0:
                    turtle.write(' ', move=True, font=font)
                turtle.write(w, move=True, font=font)
                nxt = self.cturtle.position()[0]
                mid = prv + (nxt - prv) / 2
                width = nxt-prv
                prv = nxt
                yield dict(mid=mid, nxt=nxt, width=width)

        s, ts = convert_binary_bracketing(tree_to_string(parse))

        lst = list(write_sentence(tokens))
        mids = [d['mid'] for d in lst]
        nxts = [d['nxt'] for d in lst]
        ws = [d['width'] for d in lst]
        # x0 = nxts[0] - ws[0]
        n = len(mids)
        xMax = max(mids)

        self.srdraw(s, ts, ws, x0=x0, style=style)

        box = self.cturtle.bounding_box()

        return box


def run_one(options, data, tokens, name, tree_key='binary_tree'):
    example_id = data['example_id']
    parse = data[tree_key]

    style = data.get('style', None)

    with tempfile.NamedTemporaryFile(mode='w') as f:
        path_ps = f.name
        path_pdf = os.path.join(options.out_dir, '{}-{}.pdf'.format(name, example_id))

        turtle.speed('fastest')

        fig = TreeFig(color=options.color, size=options.size)

        # Setup
        scale = 1
        # x0 = -300 * scale
        y0 = 65 * scale
        yMax = 200 * scale
        widthWindow = WIDTH * scale
        heightWindow = 500 * scale
        x0 = -widthWindow/2 + 10

        fig.setup_turtle(widthWindow, heightWindow, scale, x0)

        # Init turtle.
        ts = turtle.getscreen()
        ts.tracer(0, 0) # https://stackoverflow.com/questions/16119991/how-to-speed-up-pythons-turtle-function-and-stop-it-freezing-at-the-end

        # Draw settings.
        settings = {}
        settings['style'] = style

        # Draw.
        bounding_box = fig.draw_tree(parse, tokens, **settings)

        # Update Canvas.
        ts.update()
        ts.getcanvas().postscript(file=path_ps)

        print('writing to {}'.format(path_pdf))

        os.system('ps2pdf -dEPSCrop {} {}'.format(path_ps, path_pdf))

        # Crop the image.

        # print('bounding box = {}'.format(bounding_box))

        output_filename = os.path.join(
            options.out_dir, '{}-{}-cropped.pdf'.format(name, example_id))
        input1 = PdfFileReader(open(path_pdf, "rb"))
        output = PdfFileWriter()

        page = input1.getPage(0)
        # print('mediaBox', page.mediaBox)
        # print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
        page.trimBox.lowerLeft = (bounding_box['x0'], bounding_box['y0'])
        page.trimBox.upperRight = (bounding_box['x1'], bounding_box['y1'])
        page.cropBox.lowerLeft = (bounding_box['x0'], bounding_box['y0'])
        page.cropBox.upperRight = (bounding_box['x1'], bounding_box['y1'])
        output.addPage(page)

        print('writing to {}'.format(output_filename))
        outputStream = open(output_filename, "wb")
        output.write(outputStream)
        outputStream.close()

    return bounding_box


def run(options):
    inp_file = [os.path.expanduser(x) for x in options.inp_file.split(',')]
    prefix = options.prefix.split(',')
    tree_key_lst = options.tree_key.split(',')

    assert len(inp_file) == len(prefix), (inp_file, prefix)
    assert len(inp_file) == len(tree_key_lst), (inp_file, tree_key_lst)

    ################################################################################
    # Local variables.
    ################################################################################

    chosen = []
    boxes = {}

    ################################################################################
    # Read input files.
    ################################################################################

    name2table = {name: {} for name in prefix}

    first_table = None

    for i, filename in enumerate(inp_file):
        name = prefix[i]
        tree_key = tree_key_lst[i]
        table = name2table[name]
        if first_table is None:
            first_table = table

        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                data = json.loads(line)
                data['name'] = name
                table[data['example_id']] = data

    id_lst = options.tree_ids.split(',') \
        if options.tree_ids is not None \
        else list(first_table.keys())

    ################################################################################
    # Create PDFs for each tree.
    ################################################################################

    for i, name in enumerate(prefix):
        tree_key = tree_key_lst[i]
        table = name2table[name]

        for example_id in id_lst:
            data = table[example_id]

            # Get tokens.
            parse = table[example_id][tree_key]
            tokens = flatten_tree(parse)

            # Debug.
            print('tokens = {}'.format(tokens))
            print('template = {}'.format([{} for _ in tokens]))

            # Run.
            bounding_box = run_one(options, data, tokens, name, tree_key)

            # For log file.
            chosen.append(data)

            boxes[(name, example_id)] = bounding_box

    ################################################################################
    # Stitch PDFs together.
    ################################################################################

    if len(prefix) > 1:
        for example_id in id_lst:
            output_lst = []
            for name in prefix:
                output_filename = os.path.join(
                    options.out_dir, '{}-{}-cropped.pdf'.format(name, example_id))
                output_lst.append(output_filename)

            reader_lst = [PdfFileReader(open(fn, "rb")) for fn in output_lst]
            output = PdfFileWriter()

            x = reader_lst[0]
            print(example_id)
            print(x.pages[0].mediaBox.getHeight())
            print(x.pages[0].trimBox.getHeight())
            print(x.pages[0].cropBox.getHeight())

            total_height = sum([x.pages[0].trimBox.getHeight() for x in reader_lst])
            max_width = max([x.pages[0].trimBox.getWidth() for x in reader_lst])

            page = output.addBlankPage(
                width=max_width,
                height=total_height,
            )

            import decimal
            sofar = total_height
            for i, (x, name) in enumerate(zip(reader_lst, prefix)):
                bounding_box = boxes[(name, example_id)]

                pdf_height = x.pages[0].mediaBox.getHeight()
                crop_height = decimal.Decimal(bounding_box['y1']) \
                    - decimal.Decimal(bounding_box['y0'])
                pad_top = pdf_height - decimal.Decimal(bounding_box['y1'])
                pad_bottom = decimal.Decimal(bounding_box['y0'])

                ystart = sofar - (pad_bottom + crop_height)

                page.mergeTranslatedPage(x.pages[0], 0, ystart)

                sofar = sofar - crop_height

            # TODO: Do we need a unifying name?
            output_filename = os.path.join(
                options.out_dir, '{}-collate.pdf'.format(example_id))
            outputStream = open(output_filename, "wb")
            output.write(outputStream)
            outputStream.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_file', default='examples/examples.jsonl', type=str,
                        help='Path to input file. You may add multiple files as a'
                             'comma-delimited list, and the outputs will be'
                             'stitched together.')
    parser.add_argument('--tree_ids', default=None, type=str,
                        help='Comma-delimited list of ids to use. If `None` then'
                             'all ids found will be used.')
    parser.add_argument('--tree_key', default='tree', type=str,
                        help='Key for tree. Supports multiple'
                             'entries by comma-delimited list, and should be same'
                             'length as `--inp_file`.')
    parser.add_argument('--prefix', default='tt', type=str,
                        help='Prefix added to output file names. Supports multiple'
                             'entries by comma-delimited list, and should be same'
                             'length as `--inp_file`.')
    parser.add_argument('--out_dir', default='examples/out', type=str,
                        help='Output files will be saved in this directory.')
    parser.add_argument('--color', default='#000', type=str)
    parser.add_argument('--size', default=None, type=float)
    options = parser.parse_args()

    run(options)
