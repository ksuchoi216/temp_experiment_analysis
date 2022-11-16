
import os
import sys
import subprocess
import numpy as np


# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
if sys.platform.lower() == "win32":
    os.system('color')
        
class ColorCode:
    
    # predefined set
    
    red = '\x1b[1;31m'
    green = '\x1b[1;32m'
    yellow = '\x1b[1;33;47m'
    blue = '\x1b[1;34m'
    magenta = '\x1b[1;35m'
    cyan = '\x1b[1;36m'
    
    # for custom color
    
    foreground = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'gray': 37  # originally white?
    }
    
    background = { 
        'black': 100,
        'red': 101,
        'green': 102,
        'yellow': 103,
        'blue': 104,
        'magenta': 105,
        'cyan': 106,
        'gray': 107
    }
    
    style = {
        'default': 0,
        'bold': 1,
        'underline': 4
    }

    def __init__(self, fg_color, bg_color='white', style='default'):
        if fg_color in self.foreground:
            self.fg = self.foreground[fg_color]
        else:
            self.fg = 30
        if bg_color in self.background:
            self.bg = self.background[bg_color]
        else:
            self.bg = 0
        if style in self.style:
            self.style = self.style[style]
        else:
            self.style = 0
        # print(self.__dict__)
        
    def __str__(self):
        # return f'\x1b[1;91m \x1b[49m'
        if self.bg > 0:
            return f'\x1b[{self.style};{self.fg}m\x1b[{self.bg}m'
        else:
            return f'\x1b[{self.style};{self.fg}m'


def printc(*args, **kargs):
    """print arguemtns with specied color.
    
    Args:
        color: printing foreground color specified in ColorCode.foreground
        background: printing background color specified in ColorCode.background
        style: printing text style specified in ColorCode.style
        
    Returns:
        None

    Raise:
        None
    """
    if 'color' in kargs:
        if kargs['color'] is not None:
            color = kargs['color'].casefold()
        else:
            color = 'black'
        del kargs['color']
    else:
        color = 'black'
    if 'background' in kargs:
        background = kargs['background'].casefold()
        del kargs['background']
    else:
        background = 'white'
    if 'style' in kargs:
        style = kargs['style']
        del kargs['style']
    else:
        style = 'default'

    if 'end' in kargs:
        # __end = kargs['end'] if 'end' in kargs else '\n'
        __end = kargs['end']
        del kargs['end']
    else:
        __end = '\n'
    
    print(ColorCode(color, background, style), end='')
    print(*args, **kargs, end='')
    print('\x1b[0m', end=__end)


## backward-compatibility supports
    
def colored_text(color, text):
    """Wrapper for printc. Just for backward compatibility.
    
    Args:
        color: printing foreground color specified in ColorCode.foreground
        text: text to print
    """
    printc(text, color=color)

    
def info(text, data=None, desc=None):
    """Wrapper for printc. Just for backward compatibility.
    
    Args:
        text: text to print
        data: data to print
        desc: description to print
    """    
    printc(text, color='blue', style='bold')
    if desc:
        printc(desc, color='green')
    if data:
        print("{}\n".format(data))

        
def alert(text, desc=None):
    """Wrapper for printc. Just for backward compatibility.
    
    Args:
        text: text to print
        desc: description to print
    """    
    printc(text, color='red', style='bold')
    if desc:
        printc(desc, color='green')


def debug(*args, **kwargs):
    """Wrapper for printc. Just for backward compatibility.
    
    Args:
        text: text to print
        desc: description to print
    """
    if 'DEBUG' in os.environ and \
        os.environ['DEBUG'].lower() in ['true', '1', 't', 'y', 'yes']:
        printc(*args, **kwargs, color='green')


def red_info(title, content):
    """Redundant but for backward compatibility.
    
    Args:
        title: title text
        content: content text
    """
    print("\x1b[1;91m{}\x1b[0m".format(title), "\x1b[30m{}\x1b[0m".format(content)) 

    
def green_info(title, content):
    """Redundant but for backward compatibility.
    
    Args:
        title: title text
        content: content text
    """
    print("\x1b[1;32m{}\x1b[0m".format(title), "\x1b[30m{}\x1b[0m".format(content))


# https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
def print_dict(dictionary):
    # pp(dictionary)
    print("\n".join("{}\t{}".format(k, v) for k, v in dictionary.items()))
